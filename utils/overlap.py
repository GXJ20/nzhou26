# %%
def overlap_to_acc(total, pred, true, overlap):
    true_pos = overlap
    false_pos = pred - overlap
    false_neg = true - overlap
    true_neg = total - (pred + true -overlap)
    acc = (true_pos+true_neg) / total
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1 = 2 * ((precision*recall)/(precision+recall))
    return acc, f1
overlap_to_acc(total=499264, pred=299559, true=304549, overlap=203347)

# %%
