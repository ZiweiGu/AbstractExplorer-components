
import nltk
import spacy
import string
import json
import re
import ast
import openai
from promptengine.pipelines import PromptPipeline
from promptengine.template import PromptTemplate, PromptPermutationGenerator
from promptengine.utils import LLM, extract_responses

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

openai.api_key = 'YOUR_KEY_HERE'

TEMPERATURE = 0.1 #The temperature for ChatGPT calls

SENTENCE_CLASSIFIER_PROMPT_TEMPLATE = \
"""The following sentence is from a paper abstract. Please classify it into one of the five categories below.
"${sentence}"

Categories: 1 Problem Domain (Introduction of the problem area), 2 Gaps in Prior Work, 3 Methodology (Work done by the authors), 4 Results & Findings, 5 Conclusion (Or implications for future work)

Please answer only the category number and name."""

PHRASE_SPLITTER_PROMPT_TEMPLATE = \
"""Does the following sentence end properly? Do not consider punctuation.  
"${sentence}"

Please answer only Yes or No."""

PHRASE_TAGGER_PROMPT_TEMPLATE = \
"""A sentence (from a paper abstract) was splitted into several segments, put into the following list. For each list element, please classify it into one of the 9 categories below, based on what it describes.
"${sentence}"

Categories: 
0 Status Quo/Context (the particular context or existing work)
1 Challenge/Problem/Obstacle (often starts with 'however', gaps in prior work)
2 Contribution (what the authors did)
3 Purpose/Goal/Focus (why the work was done)
4 Methodology (how the work was done)
5 Participants (who were involved)
6 System Description (of a system the authors developed or proposed)
7 Findings
8 Example

Please return a python list of the category numbers only. The length of that list must be the same as that of the input list. If the task is impossible, return an empty list."""


api_call_counter = 0

class SentenceClassifierPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(SENTENCE_CLASSIFIER_PROMPT_TEMPLATE)
        storageFile = 'sentence_classifier_responses.json'
        super().__init__(storageFile)
    def gen_prompts(self, properties):
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "sentence": properties["sentence"]
        }))
    
class PhraseSplitterPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(PHRASE_SPLITTER_PROMPT_TEMPLATE)
        storageFile = 'phrase_splitter_responses.json'
        super().__init__(storageFile)
    def gen_prompts(self, properties):
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "sentence": properties["sentence"]
        }))

class Phrase_TaggerPromptPipeline(PromptPipeline):
    def __init__(self):
        self._template = PromptTemplate(PHRASE_TAGGER_PROMPT_TEMPLATE)
        storageFile = 'phrase_tagger_responses.json'
        super().__init__(storageFile)
    def gen_prompts(self, properties):
        gen_prompts = PromptPermutationGenerator(self._template)
        return list(gen_prompts({
            "sentence": properties["sentence"]
        }))

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
    
def split_and_concatenate(sentence):
    # Remove punctuation at the end of the sentence
    result = []
    # sentence = sentence.rstrip(string.punctuation)
    # Split the sentence into words
    tokens = []
    doc = nlp(sentence.strip())
    for token in doc:
        if token.is_punct:
            # Append punctuation to the preceding token (if exists)
            if tokens:
                tokens[-1] += token.text
        else:
            tokens.append(token.text)
        if token.head.i > token.i: # current word has a parent on the right of it, so ending the sentence here will not make sense.
            continue
        if token.i + 1 < len(doc) and doc[token.i + 1].text.lower() == "and":
            continue  # Skip concatenating if the next token is "and"
        result.append((' '.join(tokens)))     
    return result



def strip_wrapping_quotes(s: str) -> str:
    if s[0] == '"': s = s[1:]
    if s[-1] == '"': s = s[0:-1]
    return s

def extract_new_phrases(sentences):
    new_phrases = []
    previous_sentence = ""
    for sentence in sentences:
        # Find the part of the sentence that is new compared to the previous one
        if sentence.startswith(previous_sentence):
            new_phrase = sentence[len(previous_sentence):]
        else:
            new_phrase = sentence
        new_phrases.append(new_phrase)
        previous_sentence = sentence
    return new_phrases
        
sentence_classifier = SentenceClassifierPromptPipeline()
all_data = prepare_paper_data(0, 8)
group1 = []
group2 = []
group3 = []
group4 = []
group5 = []
counter_1 = 0
counter_2 = 0
counter_3 = 0
counter_4 = 0
counter_5 = 0
for d in all_data:
    responses = []
    sentence_classifier.clear_cached_responses()
    api_call_counter += 1
    for res in sentence_classifier.gen_responses({"sentence": d['sentence']}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
        responses.extend(extract_responses(res, llm=LLM.ChatGPT))
    if '1' in responses[0]:
        group1.append({'sentence_id': counter_1, 'paper_id': d['paper_id'], 'authors': d['authors'], 'sentence': d['sentence'], 'segments': []})
        counter_1 += 1
    elif '2' in responses[0]:
        group2.append({'sentence_id': counter_2, 'paper_id': d['paper_id'], 'authors': d['authors'], 'sentence': d['sentence'], 'segments': []})
        counter_2 += 1
    elif '3' in responses[0]:
        group3.append({'sentence_id': counter_3, 'paper_id': d['paper_id'], 'authors': d['authors'], 'sentence': d['sentence'], 'segments': []})
        counter_3 += 1
    elif '4' in responses[0]:
        group4.append({'sentence_id': counter_4, 'paper_id': d['paper_id'], 'authors': d['authors'], 'sentence': d['sentence'], 'segments': []})
        counter_4 += 1
    else:
        print('Classified to group 5 because  ' + responses[0])
        group5.append({'sentence_id': counter_5, 'paper_id': d['paper_id'], 'authors': d['authors'], 'sentence': d['sentence'], 'segments': []})
        counter_5 += 1

print('Sentence Classification Complete.')
        
buggy_case_counter = 0
phrase_splitter = PhraseSplitterPromptPipeline()
phrase_tagger = Phrase_TaggerPromptPipeline()
groups = [group1, group2, group3, group4, group5]
for i, group in enumerate(groups):
    for d in group:
        candidates = split_and_concatenate(d['sentence'])
        result = []
        for candidate in candidates:
            responses = []
            phrase_splitter.clear_cached_responses()
            api_call_counter += 1
            for res in phrase_splitter.gen_responses({"sentence": candidate}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
                responses.extend(extract_responses(res, llm=LLM.ChatGPT))
            responses = [strip_wrapping_quotes(r) for r in responses]
            if 'yes' in responses[0].lower():
                result.append(candidate)
        segmented_sent = extract_new_phrases(result)
        if segmented_sent == []:
            segmented_sent = [d['sentence']]
        tmp = []
        phrase_tagger.clear_cached_responses()
        api_call_counter += 1
        for res in phrase_tagger.gen_responses({"sentence": str(segmented_sent)}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
            tmp.extend(extract_responses(res, llm=LLM.ChatGPT))
        color_list = ast.literal_eval(tmp[0])
        if len(color_list) == len(segmented_sent):
            for j, segment in enumerate(segmented_sent):
                d['segments'].append({'text': segment, 'color': color_list[j]})
        else:
            buggy_case_counter += 1
            print(segmented_sent)
            print(color_list)
            for j, segment in enumerate(segmented_sent):
                d['segments'].append({'text': segment, 'color': 0})

    with open('group'+ str(i+1)+'.json', 'w') as f:
        json.dump(group, f)
    print('Group file generated.')


print('Total number of OpenAI API calls is ' + str(api_call_counter))

                


