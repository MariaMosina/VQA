!nvidia-smi
import torch
torch.cuda.is_available()

from pathlib import Path
import json
import pandas as pd
from torch.utils.data import Dataset
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import numpy

!wget http://images.cocodataset.org/zips/train2014.zip
!unzip train2014.zip
!wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
!unzip v2_Questions_Train_mscoco.zip
!wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
!unzip v2_Annotations_Train_mscoco.zip

Train_DIR_images = Path("train2014")
train_images = sorted(list(Train_DIR_images.rglob('*.jpg')))

with open("v2_OpenEnded_mscoco_train2014_questions.json", 'r') as f:
    questions = json.loads(f.read())
train_questions = questions['questions']

with open("v2_mscoco_train2014_annotations.json", 'r') as f:
    annotations = json.loads(f.read())
train_annotations = annotations['annotations']
for i in range(len(train_annotations)):
  for j in ['answer_type', 'multiple_choice_answer', 'question_type']:
    del train_annotations[i][j]
    
dictionary_que_ans = []
sw = stopwords.words('english')
for i in range(len(train_questions)):
  dictionary_que_ans.append(train_questions[i]['question'])
  dictionary_que_ans[i] = word_tokenize(dictionary_que_ans[i])
  dictionary_que_ans[i] = [word for word in dictionary_que_ans[i] if word not in sw]
ln = len(dictionary_que_ans)
for i in range(len(train_annotations)):
  for j in range(10):
    dictionary_que_ans.append(train_annotations[i]['answers'][j]['answer'])
    dictionary_que_ans[ln + i*10 + j] = word_tokenize(dictionary_que_ans[ln + i*10 + j])
    dictionary_que_ans[ln + i*10 + j] = [word for word in dictionary_que_ans[ln + i*10 + j] if word not in sw]
    
for i in range(len(train_questions)):
  train_questions[i]['question'] = dictionary_que_ans[i]
for i in range(len(train_annotations)):
  for j in range(10):
    train_annotations[i]['answers'][j]['answer'] = dictionary_que_ans[ln + i*10 + j]
    
!wget https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
!gunzip GoogleNews-vectors-negative300-SLIM.bin.gz
model_vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin", binary=True)

for i in range(len(train_questions)):
  for j in range(len(train_questions[i]['question'])):
    train_questions[i]['question'][j] = model_vec[train_questions[i]['question'][j]]
    
mean_sent = []
for k in range(len(train_questions)):
  b = []
  for i in range(300):
    s = 0
    for j in range(len(train_questions[k]['question'])):
      s += train_questions[k]['question'][j][i]
    s  = s/len(train_questions[k]['question'])  
    b.append(s)
  mean_sent.append(b)
  
for i in range(len(train_questions)):
  train_questions[i]['question'] = mean_sent[i]
  
for i in range(len(train_annotations)):
  for i in range(10):
    train_annotations[i]['answers'][j]['answer'] = model_vec[train_annotations[i]['answers'][j]['answer']]
    
DATA_MODES = ['train', 'val', 'test']
DEVICE = torch.device("cuda")

class VQADataset(Dataset):
    def __init__(self, images, questions, answers=None, mode = 'start', shape = 0):
      super().__init__()
      self.images = sorted(images)
      self.images_id = [path.name for path in self.images][-6:]
      self.questions = questions
      self.question_id = []
      for i in range(len(self.questions)):
        self.question_id.append(self.questions[i]['question_id'])
      self.mode = mode
      
      if self.mode not in DATA_MODES:
        print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
        raise NameError
      
      if shape != 0:
        self.len_ = shape
      else:
        self.len_ = len(self.questions) * 10
      
      if self.mode != 'test':
        self.answers = answers
        self.answer_id = []
        for i in range(len(self.answers)):
          for j in range(10):
            if self.answers[i]["answers"][j]["answer_confidence"] == 'yes' or self.answers[i]["answers"][j]["answer_confidence"] == 'maybe':
              self.answer_id.append(self.answers[i]['question_id']*10 + self.answers[i]['answers'][j]['answer_id'])
        
        
          
    def __len__(self):
      return self.len_
    
    def load_image(self, file):
      image = Image.open(file)
      image.load()
      return image
    
    def _prepare_sample(self, image):
        image = image.resize((224, 224))
        return np.array(image)
    
    def __getitem__(self, index):
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])
      im = self.load_image(self.images[index//10000])
      im = self._prepare_sample(im)
      im = np.array(im/255, dtype='float32')
      im = transform(im)
      
      
      
      if self.mode == 'test':
        return im
