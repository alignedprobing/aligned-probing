from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

class TaskEncoder:

    def get_instruction(self, task_types, entry, demonstration = False, template_index=0):
        pass


    def evaluate_generated_text(self, generated_text, truth_text):
        rouge = Rouge()
        rouge_scores = rouge.get_scores(generated_text, truth_text)[0]

        return {
            "METERO": meteor_score([truth_text.split()], generated_text.split()),
            "BLEU": sentence_bleu([truth_text.split()], generated_text.split()),
            "ROUGE-1": rouge_scores["rouge-1"]["f"],
            "ROUGE-2": rouge_scores["rouge-2"]["f"],
            "ROUGE-L": rouge_scores["rouge-l"]["f"]
        }