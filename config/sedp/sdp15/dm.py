import re
import json
from transformers.data import metrics
from tqdm import tqdm


test_split = 1410
split_token = '<fun_spt>'
dep_token = '['

special_tokens = [split_token, '[pid_token]', '[dep_token]',
    '[acomp]', '[advcl]', '[advmod]', '[amod]', '[appos]', '[aux]', '[auxpass]', '[cc]', '[ccomp]', '[conj]', '[cop]', '[csubj]', '[csubjpass]', '[dep]', '[det]', '[discourse]', '[dobj]', '[expl]', '[iobj]', '[mark]', '[mwe]', '[neg]', '[nn]', '[npadvmod]', '[nsubj]', '[nsubjpass]', '[num]', '[number]', '[parataxis]', '[pcomp]', '[pobj]', '[poss]', '[possessive]', '[preconj]', '[predet]', '[prep]', '[prt]', '[punct]', '[quantmod]', '[rcmod]', '[root]', '[tmod]', '[vmod]', '[xcomp]',
    '[ACMP-arg]', '[ACMP]', '[ACT-arg]', '[ADDR-arg]', '[ADVS.member]', '[AIM-arg]', '[AIM]', '[APPS.member]', '[APP]', '[ARG1]', '[ARG2]', '[ARG3]', '[ARG4]', '[ATT]', '[AUTH]', '[BEN-arg]', '[BEN]', '[BV]', '[CAUS-arg]', '[CAUS]', '[CM]', '[CNCS]', '[COMPL-arg]', '[COMPL]', '[COND]', '[CONFR.member]', '[CONJ.member]', '[CONTRA.member]', '[CONTRD]', '[CPHR-arg]', '[CPHR]', '[CPR-arg]', '[CPR]', '[CRIT-arg]', '[CRIT]', '[CSQ.member]', '[DESCR]', '[DIFF-arg]', '[DIFF]', '[DIR1-arg]', '[DIR1]', '[DIR2-arg]', '[DIR2]', '[DIR3-arg]', '[DIR3]', '[DISJ.member]', '[DPHR-arg]', '[DPHR]', '[EFF-arg]', '[EXT-arg]', '[EXT]', '[GRAD.member]', '[HER]', '[ID]', '[INTF]', '[INTT-arg]', '[INTT]', '[LOC-arg]', '[LOC]', '[MANN-arg]', '[MANN]', '[MAT]', '[MEANS-arg]', '[MEANS]', '[MOD]', '[NE.member]', '[NE]', '[OPER.member]', '[ORIG-arg]', '[PARTL]', '[PAR]', '[PAT-arg]', '[PREC]', '[REAS.member]', '[REG]', '[RESL]', '[RESTR]', '[RHEM]', '[RSTR]', '[SM]', '[SUBS]', '[TFHL-arg]', '[TFHL]', '[TFRWH]', '[THL-arg]', '[THL]', '[THO]', '[TOWH-arg]', '[TOWH]', '[TPAR]', '[TSIN-arg]', '[TSIN]', '[TTILL]', '[TWHEN-arg]', '[TWHEN]', '[VOCAT]', '[_after_c]', '[_and+also_c]', '[_and+not_c]', '[_and+so_c]', '[_and+then_c]', '[_and+thus_c]', '[_and+yet_c]', '[_and_c]', '[_as+well+as_c]', '[_but+also_c]', '[_but+not_c]', '[_but_c]', '[_even_c]', '[_except+that_c]', '[_except_c]', '[_formerly_c]', '[_if+not_c]', '[_instead+of_c]', '[_minus_c]', '[_much+less_c]', '[_nor_c]', '[_not+to+mention_c]', '[_not_c]', '[_or_c]', '[_plus_c]', '[_rather+than_c]', '[_then_c]', '[_though_c]', '[_versus_c]', '[_yet_c]', '[adj_ARG1]', '[adj_ARG2]', '[adj_MOD]', '[app_ARG1]', '[app_ARG2]', '[aux_ARG1]', '[aux_ARG2]', '[aux_MOD]', '[comp]', '[comp_ARG1]', '[comp_ARG2]', '[comp_MOD]', '[comp_enough]', '[comp_less]', '[comp_not+so]', '[comp_not+too]', '[comp_so]', '[comp_too]', '[compound]', '[conj_ARG1]', '[conj_ARG2]', '[conj_ARG3]', '[coord_ARG1]', '[coord_ARG2]', '[det_ARG1]', '[dtv_ARG2]', '[it_ARG1]', '[lgs_ARG2]', '[loc]', '[lparen_ARG1]', '[lparen_ARG2]', '[lparen_ARG3]', '[manner]', '[measure]', '[none]', '[noun_ARG1]', '[noun_ARG2]', '[of]', '[parenthetical]', '[part]', '[plus]', '[poss_ARG1]', '[poss_ARG2]', '[prep_ARG1]', '[prep_ARG2]', '[prep_ARG3]', '[prep_MOD]', '[punct_ARG1]', '[quote_ARG1]', '[quote_ARG2]', '[quote_ARG3]', '[relative_ARG1]', '[relative_ARG2]', '[subord]', '[temp]', '[than]', '[times]', '[verb_ARG1]', '[verb_ARG2]', '[verb_ARG3]', '[verb_ARG4]', '[verb_MOD]',
    '[none]', '[root]',
    '[VB]', '[NNPS]', '[VBN]', '[CD]', '[,]', '[RBS]', '[NN]', '[-LRB-]', '[TO]', '[UH]', '[VBG]', '[PDT]', '[NNP]', '[.]', '[-RRB-]', '[NNS]', '[VBD]', '[IN]', '[WP]', '[$]', '[FW]', '[VBZ]', '[WRB]', '[#]', '[``]', '[RP]', '[DT]', '[PRP]', '[:]', '[EX]', '[WDT]', '[PRP$]', '[MD]', '[CC]', '[JJR]', '[VBP]', '[LS]', '[POS]', '[JJS]', "['']", '[SYM]', '[WP$]', '[JJ]', '[RB]', '[RBR]',
    '[(]', '[)]']
    

origin_tokens = [['verb'], ['plural', 'proper', 'noun'], ['verb', 'past', 'participle'], ['cardinal', 'number'], ['superlative', 'adverb'], ['noun'], ['to'], ['interjection'], ['verb', 'gerund', 'or', 'persent', 'participle'], ['predeterminer'], ['singular', 'proper', 'noun'], ['.'], ['plural', 'noun'], ['verb', 'past', 'tense'], ['preposision', 'or', 'subordinating', 'conjunction'], ['wh-pronoun'], ['$'], ['foreign', 'word'], ['verb', '3rd', 'person', 'singular', 'present'], ['wh-ab'], ['#'], ['apostrophe'], ['particle'], ['determiner'], ['personal', 'pronoun'], [':'], ['existential', 'there'], ['wh-determiner'], ['possessive', 'pronoun'], ['modal'], ['coordinating', 'conjunction'], ['adjective', 'comparative'], ['verb', 'non-3rd', 'person', 'singular', 'present'], ['list', 'item', 'marker'], ['possessive', 'ending'], ['adjective', 'superlative'], ["''"], ['symbol'], ['possessive', 'wh-pronoun'], ['adjective'], ['adverb'], ['Adverb', 'comparative'],
['root'], ['argument', '1'], ['argument', '2'], ['argument', '3'], ['argument', '4'], ['basic', 'verb'], ['after'], ['and', 'also'], ['and', 'not'], ['and', 'so'], ['and', 'then'], ['and', 'thus'], ['and', 'yet'], ['and'], ['as', 'well', 'as'], ['but', 'also'], ['but', 'not'], ['but'], ['even'], ['except', 'that'], ['formerly'], ['if', 'not'], ['instead', 'of'], ['minus'], ['much', 'less'], ['nor'], ['not', 'to', 'mention'], ['not'], ['or'], ['plus'], ['rather', 'than'], ['then'], ['though'], ['versus'], ['yet'], ['apposition'], ['complement'], ['complement', 'enthough'], ['complement', 'less'], ['complement', 'not', 'so'], ['complement', 'not', 'too'], ['complement', 'so'], ['complement', 'too'], ['compound'], ['conjuncation'], ['discourse'], ['location'], ['manner'], ['measure'], ['multi-word', 'expression'], ['negation'], ['none'], ['of'], ['parenthetical'], ['part'], ['plus'], ['poss'], ['sub', 'ordinal'], ['temporal'], ['than'], ['times']]

target_tokens = ['[VB]', '[NNPS]', '[VBN]', '[CD]', '[RBS]', '[NN]', '[TO]', '[UH]', '[VBG]', '[PDT]', '[NNP]', '[.]', '[NNS]', '[VBD]', '[IN]', '[WP]', '[$]', '[FW]', '[VBZ]', '[WRB]', '[#]', '[``]', '[RP]', '[DT]', '[PRP]', '[:]', '[EX]', '[WDT]', '[PRP$]', '[MD]', '[CC]', '[JJR]', '[VBP]', '[LS]', '[POS]', '[JJS]', "['']", '[SYM]', '[WP$]', '[JJ]', '[RB]', '[RBR]',
'[root]', '[ARG1]', '[ARG2]', '[ARG3]', '[ARG4]', '[BV]', '[_after_c]', '[_and+also_c]', '[_and+not_c]', '[_and+so_c]', '[_and+then_c]', '[_and+thus_c]', '[_and+yet_c]', '[_and_c]', '[_as+well+as_c]', '[_but+also_c]', '[_but+not_c]', '[_but_c]', '[_even_c]', '[_except+that_c]', '[_formerly_c]', '[_if+not_c]', '[_instead+of_c]', '[_minus_c]', '[_much+less_c]', '[_nor_c]', '[_not+to+mention_c]', '[_not_c]', '[_or_c]', '[_plus_c]', '[_rather+than_c]', '[_then_c]', '[_though_c]', '[_versus_c]', '[_yet_c]', '[appos]', '[comp]', '[comp_enthough]', '[comp_less]', '[comp_not+so]', '[comp_not+too]', '[comp_so]', '[comp_too]', '[compound]', '[conj]', '[discourse]', '[loc]', '[manner]', '[measure]', '[mwe]', '[neg]', '[none]', '[of]', '[parenthetical]', '[part]', '[plus]', '[poss]', '[subord]', '[temp]', '[than]', '[times]']

def clean(seq):
    seq = seq.strip()
    seq = re.sub('<pad>', '', seq)
    seq = re.sub('</s>', '', seq)
    seq = re.sub('\n', '', seq)
    seq = re.sub(' ', '', seq)
    return seq

def metric(tokenizer, predictions, golds, best_metric, extra_labels=None):
    id_arc, ood_arc = 0, 0
    id_rel, ood_rel = 0, 0
    id_gold_total, ood_gold_total = 0, 0
    id_pred_total, ood_pred_total = 0, 0
    with open("sedp_sdp15_dm_output.txt", "w") as fw:
        for sen_idx, (pred, refe) in enumerate(zip(predictions, golds)):
            pred, refe = clean(pred), clean(refe)
            fw.write(pred + "\n")
            fw.write(refe + "\n")
            pred_split = pred.split(split_token)
            refe_split = refe.split(split_token)
            pred_len = pred.count(split_token)
            refe_len = refe.count(split_token)

            if sen_idx < test_split:
                id_gold_total += refe_len
                id_pred_total += pred_len
                id_gold_total -= refe.count('[none]')
                id_pred_total -= pred.count('[none]')
            else:
                ood_gold_total += refe_len
                ood_pred_total += pred_len
                ood_gold_total -= refe.count('[none]')
                ood_pred_total -= pred.count('[none]')

            if pred_len != 0:
                ### dictionary method
                pred_dicts, refe_dicts = {}, {}
                cur_p_word, cur_r_word = '', ''
                cur_p_count, cur_r_count = 1, 1
                ## build dictionary
                for i in range(pred_len):
                    if pred_split[i].split(dep_token)[0] != cur_p_word:
                        pred_dicts.update({pred_split[i].split(dep_token)[0] + '-' + str(cur_p_count): [pred_split[i]]})
                        cur_p_count += 1
                        cur_p_word = pred_split[i].split(dep_token)[0]
                    else:
                        cur_p_count -= 1
                        pred_dicts[pred_split[i].split(dep_token)[0] + '-' + str(cur_p_count)].append(pred_split[i])
                        cur_p_count += 1
                
                for i in range(refe_len):
                    if refe_split[i].split(dep_token)[0] != cur_r_word:
                        refe_dicts.update({refe_split[i].split(dep_token)[0] + '-' + str(cur_r_count): [refe_split[i]]})
                        cur_r_count += 1
                        cur_r_word = refe_split[i].split(dep_token)[0]
                    else:
                        cur_r_count -= 1
                        refe_dicts[refe_split[i].split(dep_token)[0] + '-' + str(cur_r_count)].append(refe_split[i])
                        cur_r_count += 1
                ## build keys
                pred_keys = pred_dicts.keys()
                refe_keys = refe_dicts.keys()
                keys = pred_keys & refe_keys

                if sen_idx < test_split:
                    for key in keys:
                        if len(pred_dicts[key]) == 1 and len(refe_dicts[key]) == 1:
                            if not '[none]' in refe_dicts[key][0]:
                                if pred_dicts[key] == refe_dicts[key]:
                                    id_rel += 1
                                if ']' in pred_dicts[key][0]:
                                    if pred_dicts[key][0].split(']')[1] == refe_dicts[key][0].split(']')[1]:
                                        id_arc += 1
                        else:
                            for value in pred_dicts[key]:
                                if ']' in value:
                                    value_arc = value.split(']')[1]
                                else:
                                    value_arc = value
                                for target in refe_dicts[key]:
                                    target_arc = target.split(']')[1]
                                    if value_arc == target_arc:
                                        if not '[none]' in target:
                                            id_arc += 1
                                    if value == target:
                                        if not '[none]' in target:
                                            id_rel += 1
                                        break
                else:
                    for key in keys:
                        if len(pred_dicts[key]) == 1 and len(refe_dicts[key]) == 1:
                            if not '[none]' in refe_dicts[key][0]:
                                if pred_dicts[key] == refe_dicts[key]:
                                    ood_rel += 1
                                if ']' in pred_dicts[key][0]:
                                    if pred_dicts[key][0].split(']')[1] == refe_dicts[key][0].split(']')[1]:
                                        ood_arc += 1
                        else:
                            for value in pred_dicts[key]:
                                if ']' in value:
                                    value_arc = value.split(']')[1]
                                else:
                                    value_arc = value
                                for target in refe_dicts[key]:
                                    target_arc = target.split(']')[1]
                                    if value_arc == target_arc:
                                        if not '[none]' in target:
                                            ood_arc += 1
                                    if value == target:
                                        if not '[none]' in target:
                                            ood_rel += 1
                                        break

    if id_pred_total == 0:
        id_up, id_ur, id_uf = 0, 0, 0
        id_lp, id_lr, id_lf = 0, 0, 0
    else:
        id_up = id_arc / id_pred_total * 100
        id_ur = id_arc / id_gold_total * 100
        if id_up == 0 or id_ur == 0:
            id_uf = 0
        else:
            id_uf = 2 * id_up * id_ur / (id_up + id_ur)
        id_lp = id_rel / id_pred_total * 100
        id_lr = id_rel / id_gold_total * 100
        if id_lp == 0 or id_lr == 0:
            id_lf = 0
        else:
            id_lf = 2 * id_lp * id_lr / (id_lp + id_lr)

    if ood_pred_total == 0:
        ood_up, ood_ur, ood_uf = 0, 0, 0
        ood_lp, ood_lr, ood_lf = 0, 0, 0
    else:
        ood_up = ood_arc / ood_pred_total * 100
        ood_ur = ood_arc / ood_gold_total * 100
        if ood_up == 0 or ood_ur == 0:
            ood_uf = 0
        else:
            ood_uf = 2 * ood_up * ood_ur / (ood_up + ood_ur)
        ood_lp = ood_rel / ood_pred_total * 100
        ood_lr = ood_rel / ood_gold_total * 100
        if ood_lp == 0 or ood_lr == 0:
            ood_lf = 0
        else:
            ood_lf = 2 * ood_lp * ood_lr / (ood_lp + ood_lr)

    print('id_uf=%.2f id_lf=%.2f id_up=%d/%d=%.2f id_ur=%d/%d=%.2f id_lp=%d/%d=%.2f id_lr=%d/%d=%.2f ' % (id_uf, id_lf, id_arc, id_pred_total, id_up, id_arc, id_gold_total, id_ur, id_rel, id_pred_total, id_lp, id_rel, id_gold_total, id_lr))
    print('ood_uf=%.2f ood_lf=%.2f ood_up=%d/%d=%.2f ood_ur=%d/%d=%.2f ood_lp=%d/%d=%.2f ood_lr=%d/%d=%.2f ' % (ood_uf, ood_lf, ood_arc, ood_pred_total, ood_up, ood_arc, ood_gold_total, ood_ur, ood_rel, ood_pred_total, ood_lp, ood_rel, ood_gold_total, ood_lr))
    return {
        "id_uf": id_uf,
        "id_lf": id_lf,
        "ood_uf": ood_uf,
        "ood_lf": ood_lf
    }