
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# def install_java():
#   !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
#   os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
#   !java -version
  
# install_java()
!pip install spacy
!pip install language_tool_python
!python -m spacy download en_core_web_sm
!pip install transformers
!pip install tqdm
!pip install psutil
!pip install Cython

# with open('ranking_train.jsonl', 'r', encoding='utf-8') as f:
#     train_data = [json.loads(line) for line in f.readlines()]

# with open('ranking_test.jsonl', 'r', encoding='utf-8') as f:
#     test_data = [json.loads(line) for line in f.readlines()]
    
with open('ranking_train.jsonl', 'r') as f:
    train_data = list(f)

with open('ranking_test.jsonl', 'r') as f:
    test_data = list(f)

def parse_json_line(json_line):
    json_data = json.loads(json_line)
    text = json_data['text']
    comments = json_data['comments']
    comments_text = [comment['text'] for comment in comments]
    comments_score = [comment['score'] for comment in comments]
    return pd.DataFrame({'text': [text]*5, 'comment': comments_text, 'score': comments_score})

train_df = pd.concat([parse_json_line(line) for line in train_data], ignore_index=True)
test_df = pd.concat([parse_json_line(line) for line in test_data], ignore_index=True)
# Размер обучающей и тестовой выборок
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Общая информация по датафреймам
print(train_df.info())
print(test_df.info())

# Статистические характеристики данных
print(train_df.describe())