from src.llm_reader.LLMReader import LLMReader


class QueryAugmenter:
    def __init__(self, model_name=None, model_type=None, api_key=None, base_url=None):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.base_url = base_url
        self.client = LLMReader(model_name, model_type, api_key, base_url)
        self.rewrite_prompt = ""
        self.expand_prompt = ""
        self.load_prompt()

    def load_prompt(self):
        with open("src/prompts/query_rewriter.txt", "r") as fi:
            for line in fi.readlines():
                self.rewrite_prompt += line
        with open("src/prompts/query_expander.txt", "r") as fi:
            for line in fi.readlines():
                self.expand_prompt += line

    def rewrite(self, query):
        curr_prompt = self.rewrite_prompt.replace("{A}", query)
        output = self.client.inference_single(curr_prompt)
        return output

    def expand(self, query):
        curr_prompt = self.expand_prompt.replace("{A}", query)
        output = self.client.inference_single(curr_prompt)
        return output


if __name__ == "__main__":
    query_augmenter = QueryAugmenter("deepseek-chat", "api", "sk-7e839aa32ffd4b6da5c3c01715b6b1f5",
                                     "https://api.deepseek.com/")
    input_query = "A 37-year-old-woman presents to her primary care physician requesting a new form of birth control. She has been utilizing oral contraceptive pills (OCPs) for the past 8 years, but asks to switch to an intrauterine device (IUD). Her vital signs are: blood pressure 118/78 mm Hg, pulse 73/min and respiratory rate 16/min. She is afebrile. Physical examination is within normal limits. Which of the following past medical history statements would make copper IUD placement contraindicated in this patient?\nOptions:\nA) Renal papillary necrosis\nB) Cholesterol embolization\nC) Eosinophilic granulomatosis with polyangiitis\nD) Polyarteritis nodosa"
    # rewrite_result = query_augmenter.rewrite(input_query)
    # print("rewrite_result", rewrite_result)
    expand_result = query_augmenter.expand(input_query)
    print("expand_result", expand_result)



