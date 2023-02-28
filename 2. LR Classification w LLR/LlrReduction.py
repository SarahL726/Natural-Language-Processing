from llr import *

class LlrReduction:
    def __init__(self, X_train, y_train, X_test):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test

    def _reduce_helper(self, X_samples, top_x1, top_x2, length):
        X_new = []
        for x in X_samples:
            cur_str = ' '.join([w for w in x.split() if w in top_x1 or w in top_x2])
            if len(cur_str.split()) < length:
                cur_str = x
            X_new.append(cur_str)
        
        return X_new

    def reduce_features(self, llr_factor, label1, label2, length):
        x1 = [self._X_train[i] for i in range(0, len(self._X_train)) if self._y_train[i] == label1]
        x2 = [self._X_train[i] for i in range(0, len(self._X_train)) if self._y_train[i] == label2]
        x1_counter = Counter(' '.join(x1).split())
        x2_counter = Counter(' '.join(x2).split())

        compare = llr_compare(x1_counter, x2_counter)

        top_x1 = {x1:y for x1,y in sorted(compare.items(), key=lambda x: (-x[1], x[0]))[:math.ceil(llr_factor*len(self._X_train))]}
        top_x2 = {x2:y for x2,y in sorted(compare.items(), key=lambda x: (x[1], x[0]))[:math.ceil(llr_factor*len(self._X_train))]}

        X_train_new = self._reduce_helper(self._X_train, top_x1, top_x2, length)
        X_test_new = self._reduce_helper(self._X_test, top_x1, top_x2, length)

        return X_train_new, X_test_new