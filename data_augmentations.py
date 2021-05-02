import pandas as pd
import nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import sys

class DataAugmentation:
    #{aug} {
    # 1: run on colab
    # 2: run on laptop
    # 3: run on ray's laptop
    # }
    def __init__(self, aug_num):
        self.PATH = None
        self.aug_num = aug_num
        print(type(self.aug_num))
        self.data = pd.read_json('yelp_review_training_dataset.jsonl',lines=True)
        if self.aug_num == '1':
            self.data = pd.read_json('/content/gdrive/MyDrive/nlp_proj/yelp_review_training_dataset.jsonl',lines=True)
            self.PATH = '/content/gdrive/MyDrive/nlp_proj/augmentations/'
        elif self.aug_num == '2':
            print("HE")
            self.PATH = '/Users/edmundwang/Desktop/cs182/nlp_proj/augmentations/'
        elif self.aug_num == '3':
            self.PATH = '/Users/raymondwang/desktop/cs182/nlp_proj/augmentations/'
        else:
            print('The number entered does not correlate to a path')

    def run(self):
        X = self.data['text'][:5]
        if self.aug_num == '1':

            aug = nac.KeyboardAug()
            augmented_text = X.apply(aug.augment)
            augmented_text.to_csv(self.PATH + 'testcolab.csv', index = False)

        elif self.aug_num == '2':

            aug_syn = naw.SynonymAug(aug_src='wordnet')
            augmented_text2 = X.apply(aug_syn.augment)
            augmented_text2.to_csv(self.PATH + 'testedmund1.csv', index=False)

            aug_rws = naw.RandomWordAug(action='swap')
            augmented_text3 = X.apply(aug_rws.augment)
            augmented_text3.to_csv(self.PATH + 'testedmund2.csv', index=False)

        elif self.aug_num == '3':

            aug_del = naw.RandomWordAug(action='delete')
            augmented_text4 = X.apply(aug_del.augment)
            augmented_text4.to_csv(self.PATH+'testray.csv', index=False)

if __name__ == "__main__":
    args = sys.argv
    aug_num = args[1]
    DataAugmentation(args[1]).run()


