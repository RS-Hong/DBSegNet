import json
import os
from copy import deepcopy

import torch
from nets.segformer_training import (Boundary_Loss, CE_Loss, Dice_loss, Focal_Loss, compute_single_loss,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


class ModelEMA:
    def __init__(self, model, decay=0.999):
        source = get_train_module(model)
        self.ema = deepcopy(source).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        source_state = get_train_module(model).state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            source_value = source_state[key].detach()
            if ema_value.is_floating_point():
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)

def get_main_output(outputs):
    if isinstance(outputs, dict):
        return outputs['out']
    return outputs

def get_train_module(model_train):
    return model_train.module if hasattr(model_train, "module") else model_train

def get_warmup_progress(epoch, warmup_epoch):
    if warmup_epoch <= 0:
        return 1.0
    return min(float(epoch + 1) / float(warmup_epoch), 1.0)

def get_tr_aux_weight(epoch, start_epoch, max_weight):
    if max_weight <= 0 or epoch + 1 < start_epoch:
        return 0.0
    warm_epoch = max(start_epoch, 1)
    return max_weight * min(float(epoch + 1 - start_epoch + 1) / float(warm_epoch), 1.0)

def add_boundary_loss(loss, outputs, pngs, use_boundary_loss, boundary_loss_weight, num_classes):
    if use_boundary_loss and boundary_loss_weight > 0 and isinstance(outputs, dict) and 'boundary' in outputs:
        loss = loss + boundary_loss_weight * Boundary_Loss(outputs['boundary'], pngs, ignore_index=num_classes)
    return loss

def assert_finite_loss(loss, epoch, iteration):
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Non-finite loss at epoch %d, iteration %d: %s" % (epoch + 1, iteration + 1, str(loss.detach()))
        )

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0, fusion_warmup_epoch=0, tr_aux_start_epoch=40, tr_aux_max_weight=0.03, use_boundary_loss=False, boundary_loss_weight=0.1, grad_clip_norm=0.0, dice_weight=1.0, label_smoothing=0.0, model_ema=None, ema_start_epoch=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0
    progress        = get_warmup_progress(epoch, fusion_warmup_epoch)
    tr_aux_weight   = get_tr_aux_weight(epoch, tr_aux_start_epoch, tr_aux_max_weight)
    train_module    = get_train_module(model_train)
    if hasattr(train_module, "set_progress"):
        train_module.set_progress(progress)

    def calc_loss(pred, pngs, labels, weights):
        return compute_single_loss(
            pred,
            pngs,
            labels,
            weights,
            num_classes,
            dice_loss,
            focal_loss,
            dice_weight=dice_weight,
            label_smoothing=label_smoothing,
        )

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = model_train(imgs)
            #----------------------#
            #   计算损失
            #----------------------#
            # if focal_loss:
            #     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            # else:
            #     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
            #
            # if dice_loss:
            #     main_dice = Dice_loss(outputs, labels)
            #     loss      = loss + main_dice
            if isinstance(outputs, dict):
                main_output = outputs['out']

                loss = calc_loss(main_output, pngs, labels, weights)

                if 'aux0' in outputs:
                    loss = loss + 0.4 * calc_loss(outputs['aux0'], pngs, labels, weights)

                if 'aux3' in outputs:
                    loss = loss + 0.4 * calc_loss(outputs['aux3'], pngs, labels, weights)

                if 'tr_aux' in outputs:
                    loss = loss + tr_aux_weight * calc_loss(outputs['tr_aux'], pngs, labels, weights)
                loss = add_boundary_loss(loss, outputs, pngs, use_boundary_loss, boundary_loss_weight, num_classes)
            else:
                main_output = outputs
                loss = calc_loss(main_output, pngs, labels, weights)

                
            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                # _f_score = f_score(outputs, labels)
                _f_score = f_score(main_output, labels)

            assert_finite_loss(loss, epoch, iteration)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            if model_ema is not None and epoch + 1 >= ema_start_epoch:
                model_ema.update(model_train)
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = model_train(imgs)
                #----------------------#
                #   计算损失
                #----------------------#
                # if focal_loss:
                #     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                # else:
                #     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                #
                # if dice_loss:
                #     main_dice = Dice_loss(outputs, labels)
                #     loss      = loss + main_dice
                if isinstance(outputs, dict):
                    main_output = outputs['out']

                    loss = calc_loss(main_output, pngs, labels, weights)

                    if 'aux0' in outputs:
                        loss = loss + 0.4 * calc_loss(outputs['aux0'], pngs, labels, weights)

                    if 'aux3' in outputs:
                        loss = loss + 0.4 * calc_loss(outputs['aux3'], pngs, labels, weights)

                    if 'tr_aux' in outputs:
                        loss = loss + tr_aux_weight * calc_loss(outputs['tr_aux'], pngs, labels, weights)
                    loss = add_boundary_loss(loss, outputs, pngs, use_boundary_loss, boundary_loss_weight, num_classes)
                else:
                    main_output = outputs
                    loss = calc_loss(main_output, pngs, labels, weights)


                with torch.no_grad():
                    #-------------------------------#
                    #   计算f_score
                    #-------------------------------#
                    # _f_score = f_score(outputs, labels)
                    _f_score = f_score(main_output, labels)

            #----------------------#
            #   反向传播
            #----------------------#
            assert_finite_loss(loss, epoch, iteration)
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if model_ema is not None and epoch + 1 >= ema_start_epoch:
                model_ema.update(model_train)
            
        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'prog'      : progress,
                                'tr_aux_w'  : tr_aux_weight,
                                'clip'      : grad_clip_norm,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    eval_model = model_ema.ema if model_ema is not None and epoch + 1 >= ema_start_epoch else model_train
    eval_model.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = eval_model(imgs)
            #----------------------#
            #   损失计算
            #----------------------#
            # if focal_loss:
            #     loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            # else:
            #     loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
            #
            # if dice_loss:
            #     main_dice = Dice_loss(outputs, labels)
            #     loss  = loss + main_dice
            if isinstance(outputs, dict):
                main_output = outputs['out']

                loss = calc_loss(main_output, pngs, labels, weights)

                if 'aux0' in outputs:
                    loss = loss + 0.4 * calc_loss(outputs['aux0'], pngs, labels, weights)

                if 'aux3' in outputs:
                    loss = loss + 0.4 * calc_loss(outputs['aux3'], pngs, labels, weights)

                if 'tr_aux' in outputs:
                    loss = loss + tr_aux_weight * calc_loss(outputs['tr_aux'], pngs, labels, weights)
                loss = add_boundary_loss(loss, outputs, pngs, use_boundary_loss, boundary_loss_weight, num_classes)
            else:
                main_output = outputs
                loss = calc_loss(main_output, pngs, labels, weights)

            #-------------------------------#
            #   计算f_score
            #-------------------------------#
            # _f_score    = f_score(outputs, labels)
            _f_score = f_score(main_output, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'bd_w'      : boundary_loss_weight if use_boundary_loss else 0.0,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        avg_train_loss = total_loss / epoch_step
        avg_val_loss = val_loss / epoch_step_val
        loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        eval_callback.on_epoch_end(epoch + 1, eval_model)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (avg_train_loss, avg_val_loss))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        save_model = model_ema.ema if model_ema is not None and epoch + 1 >= ema_start_epoch else model
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), avg_train_loss, avg_val_loss)))

        if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            best_val = {
                "best_val_loss_epoch": int(epoch + 1),
                "best_val_loss": float(avg_val_loss),
                "train_loss": float(avg_train_loss),
            }
            with open(os.path.join(save_dir, "best_val_loss.json"), "w", encoding="utf-8") as f:
                json.dump(best_val, f, indent=2, ensure_ascii=False)
            if loss_history is not None and hasattr(loss_history, "log_dir"):
                with open(os.path.join(loss_history.log_dir, "best_val_loss.json"), "w", encoding="utf-8") as f:
                    json.dump(best_val, f, indent=2, ensure_ascii=False)
            
        torch.save(save_model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
