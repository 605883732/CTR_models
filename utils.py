from __future__ import print_function

import codecs
import collections
import json
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import pandas as pd
   

def hash_batch(batch,hparams):
    """
        batch: a batch of data (batch_size * field_size)
        hparams: 超参数
        针对每个特征值做一次hash（相当于onehot）
    """
    batch=pd.DataFrame(batch)
    batch=list(batch.values)
    for b in batch:
        for i in range(len(b)):
            b[i]=abs(hash('key_'+str(i)+' value_'+str(b[i]))) % hparams.hash_ids
    return batch

def print_time(s, start_time):
  """Take a start time, print elapsed duration, and return a new time."""
  print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
  sys.stdout.flush()
  return time.time()

def print_out(s, f=None, new_line=True):
  """Similar to print but with support to flush and output to a file."""
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print(out_s, end="", file=sys.stdout)

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()

def print_step_info(prefix,epoch, global_step, info):
    print_out("%sepoch %d step %d lr %g logloss %.6f gN %.2f, %s" %
      (prefix, epoch,global_step, info["learning_rate"],
       info["train_ppl"], info["avg_grad_norm"], time.ctime())) 
    
def print_hparams(hparams, skip_patterns=None, header=None):
  """Print hparams, can skip keys based on pattern."""
  if header:
      print_out("%s" % header)
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print_out("  %s=%s" % (key, str(values[key])))
