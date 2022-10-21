import re
import json
from transformers.data import metrics
from tqdm import tqdm


split_token = '<fun_spt>'

special_tokens = [split_token, '[pid_token]', '[dep_token]',
                '[acomp]', '[advcl]', '[advmod]', '[amod]', '[appos]', '[aux]', '[auxpass]', '[cc]', '[ccomp]', '[conj]', '[cop]', '[csubj]', '[csubjpass]', '[dep]', '[det]', '[discourse]', '[dobj]', '[expl]', '[iobj]', '[mark]', '[mwe]', '[neg]', '[nn]', '[npadvmod]', '[nsubj]', '[nsubjpass]', '[num]', '[number]', '[parataxis]', '[pcomp]', '[pobj]', '[poss]', '[possessive]', '[preconj]', '[predet]', '[prep]', '[prt]', '[punct]', '[quantmod]', '[rcmod]', '[root]', '[tmod]', '[vmod]', '[xcomp]', 
                '[VB]', '[NNPS]', '[VBN]', '[CD]', '[,]', '[RBS]', '[NN]', '[-LRB-]', '[TO]', '[UH]', '[VBG]', '[PDT]', '[NNP]', '[.]', '[-RRB-]', '[NNS]', '[VBD]', '[IN]', '[WP]', '[$]', '[FW]', '[VBZ]', '[WRB]', '[#]', '[``]', '[RP]', '[DT]', '[PRP]', '[:]', '[EX]', '[WDT]', '[PRP$]', '[MD]', '[CC]', '[JJR]', '[VBP]', '[LS]', '[POS]', '[JJS]', "['']", '[SYM]', '[WP$]', '[JJ]', '[RB]', '[RBR]'
                ]

origin_tokens = [['verb'], ['plural', 'proper', 'noun'], ['verb', 'past', 'participle'], ['cardinal', 'number'], ['superlative', 'adverb'], ['noun'], ['to'], ['interjection'], ['verb', 'gerund', 'or', 'persent', 'participle'], ['predeterminer'], ['singular', 'proper', 'noun'], ['.'], ['plural', 'noun'], ['verb', 'past', 'tense'], ['preposision', 'or', 'subordinating', 'conjunction'], ['wh-pronoun'], ['$'], ['foreign', 'word'], ['verb', '3rd', 'person', 'singular', 'present'], ['wh-ab'], ['#'], ['apostrophe'], ['particle'], ['determiner'], ['personal', 'pronoun'], [':'], ['existential', 'there'], ['wh-determiner'], ['possessive', 'pronoun'], ['modal'], ['coordinating', 'conjunction'], ['adjective', 'comparative'], ['verb', 'non-3rd', 'person', 'singular', 'present'], ['list', 'item', 'marker'], ['possessive', 'ending'], ['adjective', 'superlative'], ["''"], ['symbol'], ['possessive', 'wh-pronoun'], ['adjective'], ['adverb'], ['Adverb', 'comparative'],
    ['adjectival', 'complement'], ['adverbial', 'clause', 'modifier'], ['adverbial', 'modifier'], ['appositional', 'modifier'], ['auxiliary'], ['passive', 'auxiliary'], ['coordination'], ['clausal', 'complement'], ['conjunct'], ['copula'], ['clausal', 'subject'], ['clausal', 'passive', 'subject'], ['dependent'], ['determiner'], ['direct', 'object'], ['discourse'], ['direct', 'object'], ['expletive'], ['indirect', 'object'], ['marker'], ['multi', 'word', 'expression'], ['negation', 'modifier'], ['noun', 'compound', 'modifier'], ['noun', 'phrase', 'as', 'adverbial', 'modifier'], ['nominal', 'subject'], ['passive', 'nominal', 'subject'], ['numeric', 'modifier'], ['number'], ['parataxis'], ['prepositional', 'complement'], ['object', 'of', 'preposition'], ['possession', 'modifier'], ['possessive'], ['preconjunct'], ['predeterminer'], ['prepositional', 'modifier'], ['phrasal', 'verb'], ['punctuation'], ['quantifier', 'phrase', 'modifier'], ['relative', 'clause', 'modifier'], ['root'], ['temporal', 'modifier'], ['verb', 'modifier'], ['open', 'clausal', 'complement']
]

target_tokens = ['[VB]', '[NNPS]', '[VBN]', '[CD]', '[RBS]', '[NN]', '[TO]', '[UH]', '[VBG]', '[PDT]', '[NNP]', '[.]', '[NNS]', '[VBD]', '[IN]', '[WP]', '[$]', '[FW]', '[VBZ]', '[WRB]', '[#]', '[``]', '[RP]', '[DT]', '[PRP]', '[:]', '[EX]', '[WDT]', '[PRP$]', '[MD]', '[CC]', '[JJR]', '[VBP]', '[LS]', '[POS]', '[JJS]', "['']", '[SYM]', '[WP$]', '[JJ]', '[RB]', '[RBR]', 
    '[acomp]', '[advcl]', '[advmod]', '[amod]', '[appos]', '[aux]', '[auxpass]', '[cc]', '[ccomp]', '[conj]', '[cop]', '[csubj]', '[csubjpass]', '[dep]', '[det]', '[discourse]', '[dobj]', '[expl]', '[iobj]', '[mark]', '[mwe]', '[neg]', '[nn]', '[npadvmod]', '[nsubj]', '[nsubjpass]', '[num]', '[number]', '[parataxis]', '[pcomp]', '[pobj]', '[poss]', '[possessive]', '[preconj]', '[predet]', '[prep]', '[prt]', '[punct]', '[quantmod]', '[rcmod]', '[root]', '[tmod]', '[vmod]', '[xcomp]'
]




def clean(seq):
    seq = seq.strip()
    seq = re.sub('<pad>', '', seq)
    seq = re.sub('</s>', '', seq)
    seq = re.sub('\n', '', seq)
    seq = re.sub(' ', '', seq)
    return seq

def metric(tokenizer, predictions, golds, best_metric, extra_labels=None):
    dev_arc, dev_rel, dev_total_pred, dev_total_gold = 0, 0, 0, 0
    test_arc, test_rel, test_total_pred, test_total_gold = 0, 0, 0, 0

    with open('sydp_ptb_output.txt', 'w') as fw:
        for sen_idx, (pred, gold) in enumerate(zip(predictions, golds)):
            # pred = pred[5:]
            pred, gold = clean(pred), clean(gold)
            fw.write(pred + '\n')
            fw.write(gold + '\n')
            pred_len = pred.count(split_token)
            pred_chunks = pred.split(split_token)
            gold_len = gold.count(split_token)
            gold_chunks = gold.split(split_token)

            if sen_idx < 1700:
                dev_total_gold += gold_len
                dev_total_pred += pred_len

                for idx in range(min(pred_len, gold_len)):
                    pred_chunk, gold_chunk = pred_chunks[idx], gold_chunks[idx]
                    if pred_chunk == gold_chunk:
                        dev_rel += 1
                    if ']' in pred_chunk:
                        pred_head = pred_chunk.split(']')[1]
                        gold_head = gold_chunk.split(']')[1]
                        if pred_head == gold_head:
                            dev_arc += 1

            if sen_idx >= 1700:
                test_total_gold += gold_len
                test_total_pred += pred_len

                for idx in range(min(pred_len, gold_len)):
                    pred_chunk, gold_chunk = pred_chunks[idx], gold_chunks[idx]
                    if pred_chunk == gold_chunk:
                        test_rel += 1
                    if ']' in pred_chunk:
                        pred_head = pred_chunk.split(']')[1]
                        gold_head = gold_chunk.split(']')[1]
                        if pred_head == gold_head:
                            test_arc += 1
    
    if dev_total_pred == 0 or dev_total_gold == 0:
        dev_uas, dev_las = 0, 0
    else:
        dev_uas = dev_arc / dev_total_gold * 100
        dev_las = dev_rel / dev_total_gold * 100

    if test_total_pred == 0 or test_total_gold == 0:
        test_uas, test_las = 0, 0
    else:
        test_uas = test_arc / test_total_gold * 100
        test_las = test_rel / test_total_gold * 100

    if best_metric:
        if dev_las > best_metric:
            print("current best dev_las=%.2f, test_las=%.2f" % (dev_las, test_las))
    else:
        print("current best dev_las=%.2f, test_las=%.2f" % (dev_las, test_las))
    print("dev_uas=%d/%d=%.2f dev_las=%d/%d=%.2f" % (dev_arc, dev_total_gold, dev_uas, dev_rel, dev_total_gold, dev_las))
    print("test_uas=%d/%d=%.2f test_las=%d/%d=%.2f" % (test_arc, test_total_gold, test_uas, test_rel, test_total_gold, test_las))
    return {
        "dev_uas": dev_uas,
        "dev_las": dev_las,
        "test_uas": test_uas,
        "test_las": test_las
    }