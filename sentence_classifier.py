import openai
from promptengine.pipelines import PromptPipeline
from promptengine.template import PromptTemplate, PromptPermutationGenerator
from promptengine.utils import LLM, extract_responses


TEMPERATURE = 1 #The temperature for ChatGPT calls

SENTENCE_CLASSIFIER_PROMPT_TEMPLATE = \
"""The following sentence is from a paper abstract. Please classify it into one of the five categories below.
"${sentence}"

Categories: 1 Problem Domain, 2 Gaps in Prior Work, 3 Methodology, 4 Results & Findings, 5 Discussion & Conclusion

Please answer only the category number and name."""

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

def find_sentence_category(sentence, k):
    openai.api_key = k
    sentence_classifier = SentenceClassifierPromptPipeline()
    sentence_classifier.clear_cached_responses()
    responses = []
    for res in sentence_classifier.gen_responses({"sentence": sentence}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
        responses.extend(extract_responses(res, llm=LLM.ChatGPT))
    if '1' in responses[0]:
        return 1
    elif '2' in responses[0]:
        return 2
    elif '3' in responses[0]:
        return 3
    elif '4' in responses[0]:
        return 4
    else:
        return 5