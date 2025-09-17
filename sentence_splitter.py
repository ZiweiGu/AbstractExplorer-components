import nltk
import json
import re

nltk.download('punkt')

def clean_text(text):
  tmp = text.encode('ascii', 'ignore').decode()
  return re.sub(r'[\r\n\\]', '', tmp)

def prepare_paper_data(start_index, end_index):
  with open('chiConference.json', 'r') as file:
    data = json.load(file)
  chi_2024_papers = [paper for paper in data['contents'] if paper.get('track') == "CHI 2024 Papers"]
  results = []
  for i in range(start_index, end_index): # for each paper
    paper = chi_2024_papers[i]
    paper_id = i + 1
    # title = clean_text(paper['title'])
    authors = clean_text(paper['authorNames'])
    sentences = nltk.tokenize.sent_tokenize(clean_text(paper['abstract']))
    for sent in sentences:
      results.append({
        'paper_id': paper_id,
        'authors': authors,
        'sentence': sent
      })  
  return results



