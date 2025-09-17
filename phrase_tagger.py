import openai
from promptengine.pipelines import PromptPipeline
from promptengine.template import PromptTemplate, PromptPermutationGenerator
from promptengine.utils import LLM, extract_responses


TEMPERATURE = 1 #The temperature for ChatGPT calls

PHRASE_TAGGER_PROMPT_TEMPLATE = \
"""The following list represents a sentence (from a paper abstract) splitted into several meaningful segments. For each segment, please classify it into one of the 9 categories below, based on what it describes.
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

Please return a python list of the category numbers only. The length of that list must be the same as that of the input list."""

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

def find_sentence_category(sentence, k):
    openai.api_key = k
    phrase_tagger = Phrase_TaggerPromptPipeline()
    phrase_tagger.clear_cached_responses()
    responses = []
    for res in phrase_tagger.gen_responses({"sentence": sentence}, LLM.ChatGPT, n=1, temperature=TEMPERATURE):
        responses.extend(extract_responses(res, llm=LLM.ChatGPT))
    return responses[0]