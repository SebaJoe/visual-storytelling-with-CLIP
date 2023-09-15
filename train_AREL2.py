from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import sys
import logging
import copy

import clip

#finetuned CLIP

import open_clip
import finetuner


import pickle

from _finetuner.models.builders import CLIPTextBuilder, CLIPVisionBuilder


from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import opts
from dataset import VISTDataset
import models
from log_utils import Logger
import misc.utils as utils

from eval_utils import Evaluator
import criterion
from criterion import to_contiguous
from misc.yellowfin import YFOptimizer
from train import setup_optimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Flag:
    def __init__(self, D_iters, G_iters, always=None):
        self.D_iters = D_iters
        self.G_iters = G_iters

        self.flag = "Disc"
        self.iters = self.D_iters
        self.curr = 0
        self.always = always

    def inc(self):
        self.curr += 1
        if self.curr >= self.iters and self.always is None:
            if self.flag == "Disc":
                self.flag = "Gen"
                self.iters = self.G_iters
            elif self.flag == "Gen":
                self.flag = "Disc"
                self.iters = self.D_iters
            self.curr = 0

#image_cache = {}

with open('image_cache.pkl', 'rb') as p:
    image_cache = pickle.load(p)

def form_image_groups(old_fids, dataset, clip_preprocess):
  fids = []
  for ind in old_fids:
    fid_ind = 0
    for fid in ind:
      if fid_ind >= len(fids):
        fids.append([fid])
      else:
        fids[fid_ind].append(fid)
      fid_ind += 1

  images = []
  for fid_group in fids:
    group_images = []
    for fid in fid_group:
      if fid in image_cache:
        group_images.append(image_cache[fid])
      else:
        path = '../../' + dataset.mode + '_images/' + fid + '.jpg'
        #print(path)
        if os.path.exists(path):
          image = Image.open(path).convert("RGB")
          pre_image = clip_preprocess(image)
          group_images.append(pre_image)
          image_cache[fid] = pre_image
        else:
          group_images.append(None)
          image_cache[fid] = None
    images.append(group_images)
  return images

def compute_clip_combo(images, seq_words, clip_model, gen_score, fine_tuned=False, image_encoder=None, text_encoder=None):
  sim_scores = []
  seq_i = 0
  for group in images:
    #print(group)
    if None in group:
      sim_scores.append([-1])
      seq_i += 1
      continue
    else:
      #print(group)
      image_input = torch.tensor(np.stack(group)).cuda()
      text_tokens = clip.tokenize(seq_words[seq_i]).cuda()
      with torch.no_grad():
        if fine_tuned:
            image_features = image_encoder(image_input).float()
            text_features = text_encoder(text_tokens).float()
        else:
            image_features = clip_model.encode_image(image_input).float()
            text_features = clip_model.encode_text(text_tokens).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
      similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
      seq_sim = []
      for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
          if x == y:
            seq_sim.append(similarity[x, y])
      sim_scores.append(seq_sim)
      seq_i += 1

  sim_scores_tensor = torch.zeros(gen_score.size())
  ind_i = 0
  for group in sim_scores:
    ind_y = 0
    if group[0] == -1:
      sim_scores_tensor[ind_i*opt.story_size:ind_i*opt.story_size + opt.story_size] = gen_score[ind_i*opt.story_size:ind_i*opt.story_size + opt.story_size]
      ind_y += opt.story_size
    else:
      for sco in group:
        sim_scores_tensor[ind_i*opt.story_size + ind_y] = sco.item()
        ind_y += 1
    ind_i += 1

  c = torch.stack([gen_score,sim_scores_tensor.cuda()])
  harmonic_mean = ((1/c).mean(dim=0))**(-1)
  return harmonic_mean

def train(opt):
    logger = Logger(opt)
    flag = Flag(D_iters=opt.D_iter, G_iters=opt.G_iter, always=opt.always)
    ################### set up dataset and dataloader ########################
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': False})

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.workers)
    dataset.val()
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    ##################### set up model, criterion and optimizer ######
    bad_valid = 0

    # set up evaluator
    evaluator = Evaluator(opt, 'val')

    # set up criterion
    crit = criterion.LanguageModelCriterion()
    rl_crit = criterion.ReinforceCriterion(opt, dataset)

    finetuned = True
    image_encoder = None
    text_encoder = None

    model_path = "../../scp_files/youthful-lewin/models"
    
    #load clip
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    if finetuned:
        image_encoder = CLIPVisionBuilder(descriptor='openai/clip-vit-base-patch32').build()
        image_encoder.load_state_dict(torch.load(model_path + "/clip-vision/model.pt"))
        text_encoder = CLIPTextBuilder(descriptor='openai/clip-vit-base-patch32').build()
        text_encoder.load_state_dict(torch.load(model_path + "/clip-text/model.pt"))
        image_encoder = image_encoder.to("cuda")
        text_encoder = text_encoder.to("cuda")
    else:
        clip_model.cuda().eval()
        print("yaa")
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # set up model
    model = models.setup(opt)
    model.cuda()
    disc_opt = copy.copy(opt)
    disc_opt.model = 'RewardModel'
    disc = models.setup(disc_opt)
    if os.path.exists(os.path.join(logger.log_dir, 'disc-model.pth')):
       logging.info("loading pretrained RewardModel")
       disc.load_state_dict(torch.load(os.path.join(logger.log_dir, 'disc-model.pth')))
    disc.cuda()

    # set up optimizer
    optimizer = setup_optimizer(opt, model)
    disc_optimizer = setup_optimizer(opt, disc)

    dataset.train()

    
    #i = 0
    #for iter, batch in enumerate(train_loader):
    #   old_fids = batch['flickr_ids']
    #   images = form_image_groups(old_fids, dataset, clip_preprocess)

    #with open("image_cache.pkl", "wb") as p:
    #    pickle.dump(image_cache, p)
    #print(image_cache)
    #return None
    #dataset.train()
    model.train()
    disc.train()
    ############################## training ##################################
    for epoch in range(logger.epoch_start, opt.max_epochs):
        # Assign the scheduled sampling prob

        start = time.time()
        for iter, batch in enumerate(train_loader):
            logger.iteration += 1
            torch.cuda.synchronize()

            feature_fc = Variable(batch['feature_fc']).cuda()
            target = Variable(batch['split_story']).cuda()
            
            old_fids = batch['flickr_ids']

            images = form_image_groups(old_fids, dataset, clip_preprocess)
            
            index = batch['index']

            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            if flag.flag == "Disc":
                model.eval()
                disc.train()
                if opt.decoding_method_DISC == 'sample':
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True,
                                                                pad=True)
                elif opt.decoding_method_DISC == 'greedy':
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=True, rl_training=True,
                                                                pad=True)
            else:
                model.train()
                disc.eval()
                seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True, pad=True)

            copy_seq = copy.deepcopy(seq).cpu()

            seq_words = utils.decode_story(dataset.get_vocab(), copy_seq.numpy(), during_training=True)

            seq = Variable(seq).cuda()
            mask = (seq > 0).float()
            mask = to_contiguous(
                torch.cat([Variable(mask.data.new(mask.size(0), mask.size(1), 1).fill_(1)), mask[:, :, :-1]], 2))
            normed_seq_log_probs = (seq_log_probs * mask).sum(-1) / mask.sum(-1)

            gen_score = disc(seq.view(-1, seq.size(2)), feature_fc.view(-1, feature_fc.size(2)))

            #print(seq_words)            
            #print(seq_words[0].split('.'))
            #print(0/0)
            #print(images)
            #print(0/0)

            #print("donde esta la biblioteca?")

            clip_gen_score = compute_clip_combo(images, seq_words, clip_model, gen_score, fine_tuned=finetuned, image_encoder=image_encoder, text_encoder=text_encoder)

            #print("hello?")
            #print("inter")

            if flag.flag == "Disc":
                gt_score = disc(target.view(-1, target.size(2)), feature_fc.view(-1, feature_fc.size(2)))

                copy_tar = copy.deepcopy(target).cpu()
                tar_words = utils.decode_story(dataset.get_vocab(), copy_tar.numpy(), during_training=True)
                
                clip_gt_score = compute_clip_combo(images, tar_words, clip_model, gt_score)

                loss = -torch.sum(clip_gt_score) + torch.sum(clip_gen_score)

                avg_pos_score = torch.mean(clip_gt_score)
                avg_neg_score = torch.mean(clip_gen_score)

                if logger.iteration % 50 == 0:
                    logging.info("pos reward {} neg reward {}".format(avg_pos_score.data.item(), avg_neg_score.data.item()))
                    #print("PREDICTION: ", utils.decode_story(dataset.get_vocab(), seq[:1].data)[0])
                    #print("GROUND TRUTH: ", utils.decode_story(dataset.get_vocab(), target[:1].data)[0])
            else:
                rewards = Variable(clip_gen_score.data)
                #with open("/tmp/reward.txt", "a") as f:
                #    print(" ".join(map(str, rewards.data.cpu().numpy())), file=f)
                loss, avg_score = rl_crit(seq.data, seq_log_probs, baseline, index, rewards)
                # if logger.iteration % opt.losses_log_every == 0:
                avg_pos_score = torch.mean(clip_gen_score)
                #logging.info(
                #    "average reward: {} average IRL score: {}".format(avg_score.data.item(), avg_pos_score.data.item()))

            if flag.flag == "Disc":
                loss.backward()
                nn.utils.clip_grad_norm(disc.parameters(), opt.grad_clip, norm_type=2)
                disc_optimizer.step()
            else:
                tf_loss = crit(model(feature_fc, target), target)
                #print("rl_loss / tf_loss = ", loss.item() / tf_loss.item())
                loss = opt.rl_weight * loss + (1 - opt.rl_weight) * tf_loss
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
                optimizer.step()

            train_loss = loss.data.item()
            torch.cuda.synchronize()

            # Write the training loss summary
            if logger.iteration % 100 == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)
                logging.info(
                    "Epoch {} Train {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, flag.flag,
                                                                                                  iter,
                                                                                                  len(train_loader),
                                                                                                  train_loss,
                                                                                                  time.time() - start))
                start = time.time()

            if logger.iteration % 600 == 0:
                if opt.always is None:
                    # Evaluate on validation dataset and save model for every epoch
                    val_loss, predictions, metrics = evaluator.eval_story(model, crit, dataset, val_loader, opt)
                    if opt.metric == 'XE':
                        score = -val_loss
                    else:
                        score = 10
                    logger.log_checkpoint(epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer)
                    # halve the learning rate if not improving for a long time
                    if logger.best_val_score > score:
                        bad_valid += 1
                        if bad_valid >= 10:
                            opt.learning_rate = opt.learning_rate / 2.0
                            logging.info("halve learning rate to {}".format(opt.learning_rate))
                            checkpoint_path = os.path.join(logger.log_dir, 'model-best.pth')
                            model.load_state_dict(torch.load(checkpoint_path))
                            utils.set_lr(optimizer, opt.learning_rate)  # set the decayed rate
                            bad_valid = 0
                            logging.info("bad valid : {}".format(bad_valid))
                    else:
                        logging.info("achieving best {} score: {}".format(opt.metric, score))
                        bad_valid = 0
                else:
                    torch.save(disc.state_dict(), os.path.join(logger.log_dir, 'disc-model.pth'))
            flag.inc()



def test(opt):
    logger = Logger(opt)
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.test()
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
    evaluator = Evaluator(opt, 'test')
    model = models.setup(opt)
    model.cuda()
    predictions, metrics = evaluator.test_story(model, dataset, test_loader, opt)


if __name__ == "__main__":
    opt = opts.parse_opt()

    if opt.option == 'train':
        print('Begin training:')
        train(opt)
    else:
        print('Begin testing:')
        test(opt)
