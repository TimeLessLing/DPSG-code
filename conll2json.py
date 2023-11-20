from Dataloader import read_corpus
import json

max_len = 'FULL'
min_len = 'FULL'

position_token = "<fun_pid>"
dep_token = "<fun_dep>"
split_token = '<fun_spt>'

def build_edges(sentence):
    rels = []
    edges = ''
    for dep in sentence:
        rels.append(dep.rel)
        if dep.head != 0:
            edges += dep.tag + '[' + dep.rel + ']' + str(dep.head) + split_token
        else:
            if dep.id != 0:
                # edges += dep.form + dep_token + str(dep.id)  + split_token
                edges += dep.tag + '[' + dep.rel + ']' + str(dep.id)  + split_token
    return edges, rels

def build_pair(data):
    positions = []
    sentences = []
    max_sen_len = 0
    rels = []
    arc = 0
    for sentence in data:
        edges, rel = build_edges(sentence)
        arc += len(sentence)
        rels.extend(rel)
        positions.append(edges)
        words = ''
        sen_len = 0
        for dep in sentence:
            words = words + dep.form + '[' + dep.tag + ']'+ str(dep.id) + split_token
            sen_len += 4
        sentences.append(words)
        if sen_len > max_sen_len:
            max_sen_len = sen_len
    rels = list(set(rels))
    # rels = sorted(rels)
    # print(rels)
    print('max sen length = ', max_sen_len)
    print('total arcs = ', arc)
    return sentences, positions, rels

def writejson(filename, sentences, positions):
    json_data = []
    text_len, position_len = 0, 0
    for sen_idx, (sentence, position) in enumerate(zip(sentences, positions)):
        if len(sentence) > text_len:
            text_len = len(sentence)
        if len(position) > position_len:
            position_len = len(position)
        json_data.append({'input': sentence, 'output': position})
        # if sen_idx % 100 == 0:
        # if sen_idx % 1000 == 0:
            # json_data.append({'input': sentence, 'output': position})
    print("data size: ", text_len, position_len, len(json_data))
    with open(filename, 'w') as fw:
        json.dump(json_data, fw, ensure_ascii=False)
    fw.close()


suffix = '-dpsg' + '.json'
languages = ["en_ewt-ud-", "de_gsd-ud-"]
folders = ["UD_English-EWT/", "UD_German-GSD/"]


for lang, path in zip(languages, folders):
    train_data = read_corpus(path + lang + "train.conllu")
    dev_data = read_corpus(path + lang + "dev.conllu")
    test_data = read_corpus(path + lang + "test.conllu")

    train_sentences, train_pairs, train_rels = build_pair(train_data)
    dev_sentences, dev_pairs, dev_rels = build_pair(dev_data)
    test_sentences, test_pairs, test_rels = build_pair(test_data)
    writejson(path + lang + 'train' + suffix, train_sentences, train_pairs)
    writejson(path + lang + 'dev' + suffix, dev_sentences, dev_pairs)
    writejson(path + lang + 'test' + suffix, test_sentences, test_pairs)

    rels = train_rels + dev_rels + test_rels
    rels = list(set(rels))
    rels = sorted(rels)
    rels = ['[' + rel + ']' for rel in rels]
    print(rels)

    joint_sentences = dev_sentences + test_sentences
    joint_pairs = dev_pairs + test_pairs
    print('test split = ', len(dev_sentences), len(dev_pairs), len(test_sentences), len(test_pairs))
    writejson(path + lang +'joint' + suffix, joint_sentences, joint_pairs)




