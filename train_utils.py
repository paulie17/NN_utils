import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.facecolor'] = '#ffffff'

# Utilities to train the models.

def accuracy(outputs,labels):
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item()/len(preds))

def training_step(model,batch):
  images,labels = batch
  out = model(images)
  loss = F.cross_entropy(out, labels)
  acc = accuracy(out,labels)
  return loss, acc

def validation_step(model,batch):
  images,labels = batch
  out = model(images)
  loss = F.cross_entropy(out,labels)
  acc = accuracy(out,labels)
  return {'val_loss': loss.detach(),'val_acc': acc}

def validation_epoch_end(outputs):
  batch_losses = [x['val_loss'] for x in outputs]
  epoch_loss = torch.stack(batch_losses).mean() #Combine losses
  batch_accs = [x['val_acc'] for x in outputs]
  epoch_acc = torch.stack(batch_accs).mean() #Combine accuracies
  return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch,result):
  print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, train_accuracy: {:.4f} val_loss: {:.4f}, val_acc: {:.4f}"
        .format(epoch+1, result['lrs'][-1], result['train_loss'], result['train_accuracy'], result['val_loss'], result['val_acc']))

@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [validation_step(model,batch) for batch in val_loader]
  return validation_epoch_end(outputs)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay = 0.0,
                  grad_clip = None, opt_func = torch.optim.SGD):
  torch.cuda.empty_cache()
  history = []

  # Set up custom optimizer with weight decay
  optimizer = opt_func(model.parameters(), max_lr, weight_decay = weight_decay)
  # Set up one-cycle learning rate scheduler
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                              steps_per_epoch = len(train_loader))

  for epoch in range(epochs):
    # Training phase
    model.train()
    train_losses = []
    train_accuracies = []
    lrs = []
    for batch in train_loader:
      loss, acc = training_step(model,batch)
      train_losses.append(loss)
      train_accuracies.append(acc)
      loss.backward()

      # Gradient clipping
      if grad_clip is not None:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)

      optimizer.step()
      optimizer.zero_grad()

      # Record & update learning rate
      lrs.append(get_lr(optimizer))
      sched.step()

    # Validation phase
    result = evaluate(model, val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['train_accuracy'] = torch.stack(train_accuracies).mean().item()
    result['lrs'] = lrs
    epoch_end(epoch,result)
    history.append(result)
  return history

def plot_accuracies(history):
  val_accuracies = [x['val_acc'] for x in history]
  train_accuracies = [x['train_accuracy'] for x in history]
  plt.plot(val_accuracies, '-rx')
  plt.plot(train_accuracies, '-bx')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['Validation','Training'])
  plt.title('Accuracy vs No. of epochs');

def plot_losses(history):
  train_losses = [x.get('train_loss') for x in history]
  val_losses = [x['val_loss'] for x in history]
  plt.plot(train_losses, '-bx')
  plt.plot(val_losses, '-rx')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend(['Training','Validation'])
  plt.title('Loss vs No. of epochs');

def plot_lrs(history):
  lrs = np.concatenate([x.get('lrs',[]) for x in history])
  plt.plot(lrs)
  plt.xlabel('Batch no.')
  plt.ylabel('Learning rate')
  plt.title('Learning rate vs No. of batches');

def predict_image(img, model):
  # Convert to a batch of 1
  xb = to_device(img.unsqueeze(0),device)
  # Get prediction from model
  yb = model(xb)
  # Pick index with highest probability
  _, preds = torch.max(yb,dim=1)
  # Retrieve the class label
  return train_ds.classes[preds[0].item()]
