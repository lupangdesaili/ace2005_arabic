import os
from tqdm import tqdm
import pandas as pd
import stanza
from xml.etree import ElementTree
from bs4 import BeautifulSoup
import nltk
import json
import re
from torch.nn.utils.rnn import pad_sequence
import torch

def get_data_paths(ace2005_path):
    test_files, dev_files, train_files = [], [], []
    with open('./data_list_arabic.csv', mode='r') as csv_file:
        rows = csv_file.readlines()
        for row in rows[1:]:
            items = row.replace('\n', '').split(',')
            data_type = items[0]
            name = items[1]

            path = os.path.join(ace2005_path, name)
            if data_type == 'test':
                test_files.append(path)
            elif data_type == 'dev':
                dev_files.append(path)
            elif data_type == 'train':
                train_files.append(path)
    return test_files, dev_files, train_files

class Parser:
    def __init__(self, path):
        self.path = path
        self.entity_mentions = []
        self.event_mentions = []
        self.sentences = []
        self.sgm_text = ''

        self.entity_mentions, self.event_mentions, self.relation_mentions = self.parse_xml(path + '.apf.xml')
        #self.doc = self.get_sgm_doc(path + '.sgm')
        self.sents_with_pos = self.parse_sgm(path + '.sgm')
        self.fix_wrong_position()

    @staticmethod
    def clean_text(text):
        return text.replace('\n', ' ')

    def get_data(self):
        data = []
        for sent in self.sents_with_pos:
            item = dict()

            item['sentence'] = self.clean_text(sent['text'])
            item['position'] = sent['position']
            text_position = sent['position']

            for i, s in enumerate(item['sentence']):
                if s != ' ':
                    item['position'][0] += i
                    break

            item['sentence'] = item['sentence'].strip()

            entity_map = dict()
            item['golden-entity-mentions'] = []
            item['golden-event-mentions'] = []
            item['golden-relaiton-mentions'] = []

            for entity_mention in self.entity_mentions:
                entity_position = entity_mention['position']

                if text_position[0] <= entity_position[0] and entity_position[1] <= text_position[1]:

                    item['golden-entity-mentions'].append({
                        'text': self.clean_text(entity_mention['text']),
                        'position': entity_position,
                        'entity-type': entity_mention['entity-type'],
                        'head': {
                            "text": self.clean_text(entity_mention['head']["text"]),
                            "position": entity_mention["head"]["position"]
                        },
                        "entity_id": entity_mention['entity-id']
                    })
                    entity_map[entity_mention['entity-id']] = entity_mention

            for event_mention in self.event_mentions:
                event_position = event_mention['trigger']['position']
                if text_position[0] <= event_position[0] and event_position[1] <= text_position[1]:
                    event_arguments = []
                    for argument in event_mention['arguments']:
                        try:
                            entity_type = entity_map[argument['entity-id']]['entity-type']
                        except KeyError:
                            print('[Warning] The entity in the other sentence is mentioned. This argument will be ignored.')
                            continue

                        event_arguments.append({
                            'role': argument['role'],
                            'position': argument['position'],
                            'entity-type': entity_type,
                            'text': self.clean_text(argument['text']),
                        })

                    item['golden-event-mentions'].append({
                        'trigger': event_mention['trigger'],
                        'arguments': event_arguments,
                        'position': event_position,
                        'event_type': event_mention['event_type'],
                    })
            
            for relation_mention in self.relation_mentions:
                relation_position  = relation_mention['position']
                if text_position[0] <= relation_position[0] and event_position[1] <= relation_position[1]:
                    #relation_type = relation_mention['relation_type']
                    #relation_tese = relation_mention['relation_tese']
                    relation_arguments = []
                    for role,argument in relation_mention['arguments'].items():
                        relation_arguments.append({
                            'role':role,
                            'text':argument['text'],
                            'position':argument['position'],                              ##'type':
                            'head':argument['entity-head']['text'],
                            'head-position':argument['entity-head']['position'],
                        })
                    item['golden-relaiton-mentions'].append({
                        'relation-type':relation_mention['relation_type'],
                        'relation-tense':relation_mention['tense'],
                        'position':relation_mention['position'],
                        'arguments':relation_arguments
                    })
            data.append(item)
        return data

    def find_correct_offset(self, sgm_text, start_index, text):
        offset = 0
        for i in range(0, 70):
            for j in [-1, 1]:
                offset = i * j
                if sgm_text[start_index + offset:start_index + offset + len(text)] == text:
                    return offset

        print('[Warning] fail to find offset! (start_index: {}, text: {}, path: {})'.format(start_index, text, self.path))
        return offset

    def fix_wrong_position(self):
        for entity_mention in self.entity_mentions:
            offset = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=entity_mention['position'][0],
                text=entity_mention['text'])

            entity_mention['position'][0] += offset
            entity_mention['position'][1] += offset
            entity_mention['head']["position"][0] += offset
            entity_mention['head']["position"][1] += offset

        for event_mention in self.event_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=event_mention['trigger']['position'][0],
                text=event_mention['trigger']['text'])
            event_mention['trigger']['position'][0] += offset1
            event_mention['trigger']['position'][1] += offset1

            for argument in event_mention['arguments']:
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index=argument['position'][0],
                    text=argument['text'])
                argument['position'][0] += offset2
                argument['position'][1] += offset2
        
        for relation_mention in self.relation_mentions:
            offset1 = self.find_correct_offset(
                sgm_text=self.sgm_text,
                start_index=relation_mention['position'][0],
                text=relation_mention['text'])
            relation_mention['position'][0] += offset1
            relation_mention['position'][1] += offset1
            for _,value in relation_mention['arguments'].items():
                offset2 = self.find_correct_offset(
                    sgm_text=self.sgm_text,
                    start_index= value['position'][0],
                    text=value['text'])
                value['position'][0] += offset2
                value['position'][1] += offset2
    
    def parse_sgm(self, sgm_path):
        pattern = re.compile("""\n\n|\.\.\.""")
        pattern_tashkeel = re.compile("""[ًٌٍَُِّّْ]""")
        with open(sgm_path, 'r') as f:
            soup = BeautifulSoup(f.read(), features='html.parser')
            sgm_text = soup.text
            sgm_text = sgm_text.replace("؟","?")
            if  soup.doctype.attrs['source'] == "broadcast news":
                sgm_text = sgm_text.replace(",",".")
            
            self.sgm_text = sgm_text
            doc_type = soup.doc.doctype.text.strip()

            def remove_tags(selector):
                tags = soup.findAll(selector)
                for tag in tags:
                    tag.extract()

            if doc_type == 'WEB TEXT':
                remove_tags('poster')
                remove_tags('postdate')
                remove_tags('subject')
            elif doc_type in ['CONVERSATION', 'STORY']:
                remove_tags('speaker')

            sents = []
            converted_text = soup.text
            converted_text = converted_text.replace("؟","?")
            
            for sent in nltk.sent_tokenize(converted_text):
                sents.extend(pattern.split(sent))
            sents = list(filter(lambda x: len(x) > 5, sents))
            sents = sents[1:]
            sents_with_pos = []
            last_pos = 0
            for sent in sents:
                pos = self.sgm_text.find(sent, last_pos)
                last_pos = pos
                sents_with_pos.append({
                    'text': sent,
                    'position': [pos, pos + len(sent)]
                })

            return sents_with_pos
    def get_head(self):
        self.id2head = {x['entity-id']:x['head'] for x in self.entity_mentions}
        for relation_mention in self.relation_mentions:
            for _, arg in relation_mention['arguments'].items():
                arg['entity-head'] = self.id2head[arg['entity-id']]

    def parse_xml(self, xml_path):
        entity_mentions, event_mentions, relation_mentions = [], [], []
        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        for child in root[0]:
            if child.tag == 'entity':
                entity_mentions.extend(self.parse_entity_tag(child))
            elif child.tag in ['value', 'timex2']:
                entity_mentions.extend(self.parse_value_timex_tag(child))
            elif child.tag == 'event':
                event_mentions.extend(self.parse_event_tag(child))
            elif child.tag == 'relation':
                relation_mentions.extend(self.parse_relation_tag(child))
                

        return entity_mentions, event_mentions, relation_mentions

    @staticmethod
    def parse_relation_tag(node):
        relation_mentions = []
        for child in node:
            if child.tag == 'relation_mention':
                relation_mention = dict()
                relation_mention['relation_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                relation_mention['tense'] = node.attrib['TENSE']
                relation_mention['position'] = [int(child[0][0].attrib['START']),int(child[0][0].attrib['END'])]
                relation_mention['text'] = child[0][0].text
                relation_mention['arguments'] = dict()
                for child2 in child:
                    if child2.tag == 'relation_mention_argument':
                        role = child2.attrib['ROLE']
                        entity_id = child2.attrib['REFID']
                        charset = child2[0][0]
                        arg_text = charset.text
                        arg_position = [int(charset.attrib['START']), int(charset.attrib['END'])]
                        relation_mention['arguments'][role] = {'entity-id':entity_id, 'text':arg_text, 'position':arg_position}
                relation_mentions.append(relation_mention)
        return relation_mentions

    
    @staticmethod
    def parse_entity_tag(node):
        entity_mentions = []

        for child in node:
            if child.tag != 'entity_mention':
                continue
            extent = child[0]
            head = child[1]
            charset = extent[0]
            head_charset = head[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']
            entity_mention['entity-type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
            entity_mention["head"] = {"text": head_charset.text,
                                      "position": [int(head_charset.attrib['START']), int(head_charset.attrib['END'])]}

            entity_mentions.append(entity_mention)

        return entity_mentions

    @staticmethod
    def parse_event_tag(node):
        event_mentions = []
        for child in node:
            if child.tag == 'event_mention':
                event_mention = dict()
                event_mention['event_type'] = '{}:{}'.format(node.attrib['TYPE'], node.attrib['SUBTYPE'])
                event_mention['arguments'] = []
                for child2 in child:
                    if child2.tag == 'ldc_scope':
                        charset = child2[0]
                        event_mention['text'] = charset.text
                        event_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]
                    if child2.tag == 'anchor':
                        charset = child2[0]
                        event_mention['trigger'] = {
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                        }
                    if child2.tag == 'event_mention_argument':
                        extent = child2[0]
                        charset = extent[0]
                        event_mention['arguments'].append({
                            'text': charset.text,
                            'position': [int(charset.attrib['START']), int(charset.attrib['END'])],
                            'role': child2.attrib['ROLE'],
                            'entity-id': child2.attrib['REFID'],
                        })
                event_mentions.append(event_mention)
        return event_mentions

    @staticmethod
    def parse_value_timex_tag(node):
        entity_mentions = []

        for child in node:
            extent = child[0]
            charset = extent[0]

            entity_mention = dict()
            entity_mention['entity-id'] = child.attrib['ID']

            if 'TYPE' in node.attrib:
                entity_mention['entity-type'] = node.attrib['TYPE']
            if 'SUBTYPE' in node.attrib:
                entity_mention['entity-type'] += ':{}'.format(node.attrib['SUBTYPE'])
            if child.tag == 'timex2_mention':
                entity_mention['entity-type'] = 'TIM:time'

            entity_mention['text'] = charset.text
            entity_mention['position'] = [int(charset.attrib['START']), int(charset.attrib['END'])]

            entity_mention["head"] = {"text": charset.text,
                                      "position": [int(charset.attrib['START']), int(charset.attrib['END'])]}

            entity_mentions.append(entity_mention)

        return entity_mentions


def find_sent_for_relation(relation_mentions,sents):
    for relation_mention in relation_mentions:
        relation_mention['sent'] = ''
        mention_position = relation_mention['position']
        for sent in sents:
            sent_position = sent['position']
            if mention_position[0] >= sent_position[0] and mention_position[1] <= sent_position[1]:
                relation_mention['sent'] = sent
                break
        if relation_mention['sent'] == "":
            target_sents = []
            for i,sent1 in enumerate(sents):
                sent1_position = sent1['position']
                if mention_position[0] >= sent1_position[0]:
                    target_sents.append(sent1)
                    for sent2 in sents[i:]:
                        sent2_position = sent2['position']
                        if not mention_position[1] > sent2_position[1]:
                            target_sents.append(sent2)
                    break
            if target_sents != []:
                new_text = "".join(x['text'] for x in target_sents)
                new_pos = [target_sents[0]['position'][0],target_sents[-1]['position'][-1]]
                new_sent = {"text":new_text, "position":new_pos}
                relation_mention['sent'] = new_sent
            else:
                relation_mention['sent'] = None
    return relation_mentions

def get_entity_data(entity_mentions,sents):
    datas = []
    onto_pattern = re.compile("-[0-9]+$")
    for sent in sents:
        data = {"sent":sent,"entities":[]}
        sent_position = sent['position']
        for entity_mention in entity_mentions:
            mention_position  = entity_mention['position']
            if mention_position[0] >= sent_position[0] and mention_position[1] <= sent_position[1]:
                entity_mention['pos-in-sent'] = [mention_position[i] - sent_position[0] for i in range(2)]
                entity_mention['onto_id'] = onto_pattern.sub("",entity_mention['entity-id'])
                head_position = entity_mention['head']['position']
                entity_mention['head']['pos-in-sent'] = [head_position[i] - sent_position[0] for i in range(2)]
                data['events'].append(entity_mention)
        datas.append(data)
    return datas

def get_event_data(event_mentions,sents):
    datas = []
    for sent in sents:
        data = {"sent":sent,"events":[]}
        sent_position = sent['position']
        for event_mention in event_mentions:
            mention_position  = event_mention['position']
            if mention_position[0] >= sent_position[0] and mention_position[1] <= sent_position[1]:
                event_mention['pos-in-sent'] = [mention_position[i] - sent_position[0] for i in range(2)]
                trigger = event_mention['trigger']
                trigger_position = trigger['position']
                trigger['pos-in-sent'] = [trigger_position[i] - sent_position[0] for i in range(2)]
                trigger['pos-in-mention'] = [trigger_position[i] - mention_position[0] for i in range(2)]
                for arg in event_mention['arguments']:
                    arg_position = arg['position']
                    arg['pos-in-sent'] = [arg_position[i] - sent_position[0] for i in range(2)]
                    arg['pos-in-mention'] = [arg_position[i] - mention_position[0] for i in range(2)]
                data['events'].append(event_mention)
        datas.append(data)
    return datas

def to_det_rectify(words,word_ids,word_2_token):
    det = "ال"
    to = "ل"
    pattern = re.compile(r"^{}".format(det))
    for i in range(len(words)):
        if words[i] == to and words[i+1][0:2] == det:
            if word_2_token[word_ids[i]][0] == word_2_token[word_ids[i+1]][0]:
                words[i+1] = pattern.sub("ل",words[i+1])
    return words

def preprocess_stanza_sent(sent):
    num_pattern = re.compile("[0-9]+")
    start_ent_extract = lambda x:[int(y) for y in num_pattern.findall(x)] if x != None else None
    tokens = sent.tokens
    words = sent.words
    token_ids = [list(range(token.id[0],token.id[-1]+1)) for token in tokens]
    word_ids = [word.id for word in words]
    word_2_token = {}
    for word_id in word_ids:
        for i,token_id in enumerate(token_ids):
            for j,word_id in enumerate(token_id):
                word_2_token[word_id] = (i,j)
    word_texts = [word.text for word in words]
    word_texts = to_det_rectify(word_texts, word_ids, word_2_token)
    pos = [word.upos for word in words]
    misc = []
    for word in words:
        word_misc = start_ent_extract(word.misc)
        if word_misc != None:
            misc.append(word_misc)
        else:
            token_id , w_in_token = word_2_token[word.id]
            start = sent.tokens[token_id].start_char
            end = start
            for j in range(w_in_token+1):
                if j>0:
                    start += len(sent.tokens[token_id].words[j-1].text) 
                end += len(sent.tokens[token_id].words[j].text)
                if j == w_in_token:
                    end = sent.tokens[token_id].end_char
            misc.append([start,end])
    return tokens, words, word_texts, pos, misc

def preprocess_stanza_sent_pro(sent):
    num_pattern = re.compile("[0-9]+")
    start_ent_extract = lambda x:[int(y) for y in num_pattern.findall(x)] if x != None else None
    tokens = sent.tokens
    words = sent.words
    token_ids = [list(range(token.id[0],token.id[-1]+1)) for token in tokens]
    word_ids = [word.id for word in words]
    word_2_token_id = {}
    token_2_word_id = {}
    for word_id in word_ids:
        for i,token_id in enumerate(token_ids):
            token_2_word_id[i] = []
            for j,word_id in enumerate(token_id):
                word_2_token_id[word_id] = (i,j)
                token_2_word_id[i].append(word_id)
    word_texts = [word.text for word in words]
    token_texts = [token.text for token in tokens]
    word_texts = to_det_rectify(word_texts, word_ids, word_2_token_id)
    pos = [word.upos for word in words]
    misc_t = [(token.start_char,token.end_char) for token in tokens]
    misc_w = []
    for word in words:
        word_misc = start_ent_extract(word.misc)
        if word_misc != None:
            misc_w.append(word_misc)
        else:
            token_id , w_in_token = word_2_token_id[word.id]
            start = sent.tokens[token_id].start_char
            end = start
            for j in range(w_in_token+1):
                if j>0:
                    start += len(sent.tokens[token_id].words[j-1].text) 
                end += len(sent.tokens[token_id].words[j].text)
                if j == w_in_token:
                    end = sent.tokens[token_id].end_char
            misc_w.append([start,end])
    return word_texts, token_texts, misc_w, misc_t, pos

def preprocess_stanza_doc_pro(doc):
    def Merge(dict1, dict2): 
        return(dict2.update(dict1)) 
    words = []
    tokens = []
    misc_w_all = []
    misc_t_all = []
    pos_all = []
    token_2_word_id_all = {}
    w_start = 0
    t_start = 0
    for sent in doc.sentences:
        word_texts, token_texts, misc_w, misc_t, pos, token_2_word_id = preprocess_stanza_sent_pro(sent,w_start,t_start)
        w_start = list(token_2_word_id.items())[-1][1][-1] +1
        t_start = list(token_2_word_id.items())[-1][0] + 1
        words += word_texts
        tokens += token_texts
        misc_w_all += misc_w
        misc_t_all += misc_t
        pos_all += pos
        Merge(token_2_word_id, token_2_word_id_all)
    return words, tokens, misc_w_all, misc_t_all, pos_all, token_2_word_id_all

def get_token_id_for_word(word_id,tokens):
    token_ids = [token.id for token in tokens]
    target_token_idx = None
    for i,token_id in enumerate(token_ids):
        if word_id in token_id:
            target_token_idx = i
            break
    return target_token_idx

def get_char_for_word_groups(word_idx,tokens,misc,word_ids,texts,last_char = None):
    if last_char == None:
        target_token_idx = get_token_id_for_word(word_ids[word_idx],tokens)
        start = int(tokens[target_token_idx].start_char)
        end = start + len(texts[word_idx]) - 1
    else:
        start = last_char + 1
        end = start + len(texts[word_idx]) - 1
    return start,end

def preprocess_stanza_doc(doc):
    word_texts = []
    postags = []
    miscs = []
    for sent in doc.sentences:
        _ , _, text, pos, misc = preprocess_stanza_sent(sent)
        word_texts += text
        postags += pos
        miscs += misc
    return word_texts, postags, miscs

def pos_char_2_word(ace_position,miscs):
    start = None
    end = None
    for i, misc in enumerate(miscs):
        if misc[0] == ace_position[0]:
            start = i
        if misc[-1] == ace_position[-1]+1:
            end = i
    if start == None and end != None:
        for i, misc in enumerate(miscs):
            if misc[0] <= ace_position[0] and misc[-1] == ace_position[-1]+1:
                start = i
    if start != None and end == None:
        for i, misc in enumerate(miscs):
            if misc[0] == ace_position[0] and misc[-1] >= ace_position[-1]+1:
                end = i
            else:
                misc[-1] == ace_position[-1]+1
                end = i
    if start == None and end == None:
        span = ace_position[1] - ace_position[0]
        if span <3:
            for i, misc in enumerate(miscs):
                if misc[0] <= ace_position[0] and misc[-1] >= ace_position[-1]+1 and misc[1] - misc[0] >= span:
                    end = i
    return [start, end]

def preprocess_event_data(event_data,nlp):
    doc = nlp(event_data['sent']['text'])
    sent_start = event_data['sent']['position'][0]
    word_texts, postags, miscs = preprocess_stanza_doc(doc)
    for event in event_data['events']:
        #local_start = event['position'][0]
        trigger_position = [x - sent_start for x in event['trigger']['position']]
        #trigger_position_local = [x - local_start for x in event['trigger']['position']]
        trigger_span_id = pos_char_2_word(trigger_position,miscs)
        #local_words,local_pos,local_miscs = preprocess_stanza_doc(nlp(event['text']))
        #trigger_span_id_local  =  pos_char_2_word(trigger_position_local,local_miscs)
        event['trigger']['span'] = trigger_span_id
        for argument in event['arguments']:
            argument_position = [x - sent_start for x in argument['position']]
            argument_span_id = pos_char_2_word(argument_position,miscs)
            argument['span'] = argument_span_id
    event_data['nlp'] = {"tokens":word_texts,
                            "postags":postags}
    return event_data

def preprocess_event_datas(event_datas,nlp):
    event_datas = [x for x in event_datas if not x['events'] == []]
    result = []
    for data in event_datas:
        event_data = preprocess_event_data(data, nlp)
        result.append(event_data)
    return result

def to_det_rectify(words,word_ids,word_2_token):
    det = "ال"
    to = "ل"
    pattern = re.compile(r"^{}".format(det))
    for i in range(len(words)):
        if words[i] == to and words[i+1][0:2] == det:
            if word_2_token[word_ids[i]][0] == word_2_token[word_ids[i+1]][0]:
                words[i+1] = pattern.sub("ل",words[i+1])
    return words

def preprocess_ace_sent(text, nlp):
    doc = nlp(text)
    def Merge(dict1, dict2): 
        return(dict2.update(dict1)) 
    words = []
    tokens = []
    misc_w_all = []
    misc_t_all = []
    token_2_word_id_all = {}
    w_start = 0
    t_start = 0
    for sent in doc.sentences:
        word_texts, token_texts, misc_w, misc_t, pos, token_2_word_id = preprocess_stanza_sent_pro(sent,w_start,t_start)
        w_start = list(token_2_word_id.items())[-1][1][-1] +1
        t_start = list(token_2_word_id.items())[-1][0] + 1
        words += word_texts
        tokens += token_texts
        misc_w_all += misc_w
        misc_t_all += misc_t
        Merge(token_2_word_id, token_2_word_id_all)
    return words, tokens, misc_w_all, misc_t_all, token_2_word_id_all

def match_span(ace_position,text,words, tokens, misc_w_all, misc_t_all, token_2_word_id_all):
    start = ace_position[0]
    end = ace_position[1]+1
    head = None
    tail = None
    type = "O"
    for i,(token, misc, word_id) in enumerate(zip(tokens,misc_t_all,token_2_word_id_all.values())):
        if misc[0] == start and misc[1] == end:
            head = i
            tail = i+1
            break
        if misc[0] == start and misc[1] > end:
            if words[word_id[0]] == text or token[:len(text)] == text:
                head = i
                tail = i+1
                type = 'P'
                break
        if misc[0] < start and misc[1] == end:
            if misc_w_all[word_id[-1]] == text or token[-len(text):] == text:
                head = i
                tail = i+1
                type = 'S'
                break
        if misc[0] < start and misc[1] > start:
            if text in [words[j] for j in word_id]:
                head = i
                tail = i+1
                type = "I"
                break
        if misc[0] == start:
            head = i
        if start - misc[0] == 1 and words[word_id[0]] in "و ف ك ل ب".split():
            head = i
            type = "S"
        if misc[-1] == end:
            tail = i+1
            break
    span = [head,tail]
    return span,type


def preprocess_event_data(event_data, nlp):
    sentence = event_data['sent']['text']
    words, tokens, misc_w_all, misc_t_all, token_2_word_id_all = preprocess_ace_sent(sentence, nlp)
    event_data['tokens'] = tokens
    for event in event_data['events']:
        trigger = event['trigger']
        trigger_pos = trigger['pos-in-sent']
        trigger_text = trigger['text']
        trigger['span'], trigger['span-type'] = match_span(trigger_pos, trigger_text, words, tokens, misc_w_all, misc_t_all, token_2_word_id_all)
        for arg in event['arguments']:
            arg_pos = arg['pos-in-sent']
            arg_text = arg['text']
            arg['span'], arg['span-type'] = match_span(arg_pos, arg_text, words, tokens, misc_w_all, misc_t_all, token_2_word_id_all)
    return event_data

def preprocess_stanza_sent_pro(sent, w_start = 0, t_start = 0):
    num_pattern = re.compile("[0-9]+")
    start_ent_extract = lambda x:[int(y) for y in num_pattern.findall(x)] if x != None else None
    tokens = sent.tokens
    words = sent.words
    token_ids = [list(range(token.id[0],token.id[-1]+1)) for token in tokens]
    word_ids = [word.id for word in words]
    word_2_token_id = {}
    token_2_word_id = {}
    for word_id in word_ids:
        for i,token_id in enumerate(token_ids):
            token_2_word_id[i+t_start] = []
            for j,word_id in enumerate(token_id):
                word_2_token_id[word_id] = (i,j)
                token_2_word_id[i+t_start].append(word_id-1+w_start)
    word_texts = [word.text for word in words]
    token_texts = [token.text for token in tokens]
    word_texts = to_det_rectify(word_texts, word_ids, word_2_token_id)
    pos = [word.upos for word in words]
    misc_t = [(token.start_char,token.end_char) for token in tokens]
    misc_w = []
    for word in words:
        word_misc = start_ent_extract(word.misc)
        if word_misc != None:
            misc_w.append(word_misc)
        else:
            token_id , w_in_token = word_2_token_id[word.id]
            start = sent.tokens[token_id].start_char
            end = start
            for j in range(w_in_token+1):
                if j>0:
                    start += len(sent.tokens[token_id].words[j-1].text) 
                end += len(sent.tokens[token_id].words[j].text)
                if j == w_in_token:
                    end = sent.tokens[token_id].end_char
            misc_w.append([start,end])
    return word_texts, token_texts, misc_w, misc_t, pos, token_2_word_id

def check_span(event_data):
    origin = ["أ","ة","إ"]
    to_repalce = ["ا","ت","ا"]
    pattern = re.compile(r"([^\w\s]|[ \n\t])+")
    tokens = event_data['nlp']['tokens']
    wrong = []
    for event in event_data['events']:
        for arg in event['arguments']:
            span = arg['span']
            target = "".join(x for x in tokens[span[0]:span[-1]+1])
            target = pattern.sub("",target)
            text = pattern.sub("",arg['text'])
            for o,r in zip(origin,to_repalce):
                text = text.replace(o,r)
                target = target.replace(o,r)
            if target != text and text not in target:
                arg['pass-span-check'] = False
                wrong.append((target, text))
            else:
                arg['pass-span-check'] = True
        trigger_span = event['trigger']['span']
        target = "".join(x for x in tokens[trigger_span[0]:trigger_span[-1]+1])
        target = pattern.sub("",target)
        text = pattern.sub("",event['trigger']['text'])
        for o,r in zip(origin,to_repalce):
            text = text.replace(o,r)
            target = target.replace(o,r)
        if target != text and text not in target:
            event['trigger']['pass-span-check'] = False
            wrong.append((target, text))
        else:
            event['trigger']['pass-span-check'] = True
    return wrong



def matcher(tokens, text):
    pattern_a = re.compile("[^\t\n\xad ]")
    pattern_b = re.compile("^##")
    spans = []
    pattern_tashkeel = re.compile("[ًٌٍَُِّْ]")
    tail = 0
    for i,token in enumerate(tokens):
        token = pattern_b.sub("", token)
        search_space = pattern_a.search(text[tail:])
        span1,_ = search_space.span()
        head = tail + span1
        tail = head + len(token)
        if pattern_tashkeel.search(text[head:tail+2]):
            tail += len(pattern_tashkeel.findall(text[head:tail+2]))
        spans.append((head,tail))
    return spans

def match_span(tokens, text, event_span):
    token_spans = matcher(tokens, text)
    head = None
    tail = None
    start = event_span[0]
    end = event_span[-1]+1
    for j, token_span in enumerate(token_spans):
        if token_span[0] == start:
            head = j
        if token_span[-1] == end:
            tail = j
    if head == None:
        for j, token_span in enumerate(token_spans):
            if token_span[0] - start in [-1, -2]:
                head = j
    if tail == None:
        for j, token_span in enumerate(token_spans):
            if token_span[1] - end in [1, 2]:
                tail = j
    if head == None:
        for j, token_span in enumerate(token_spans):
            if token_span[0] - start in [-3, -4]:
                head = j
    if tail == None:
        for j, token_span in enumerate(token_spans):
            if token_span[1] - end in [3, 4]:
                tail = j
    if head == None:
        for j, token_span in enumerate(token_spans):
            if token_span[0] - start in [-5, -6]:
                head = j
    if tail == None:
        for j, token_span in enumerate(token_spans):
            if token_span[1] - end in [5, 6]:
                tail = j
    span = [head,tail]
    return span

def annotate(tokens, text, label_spans, labels):
    token_spans = matcher(tokens, text)
    notes = ["O"] * len(token_spans)
    spans = []
    for i, (label, label_span) in enumerate(zip(labels,label_spans)):
        head = None
        tail = None
        start = label_span[0]
        end = label_span[-1]+1
        for j, token_span in enumerate(token_spans):
            if token_span[0] == start:
                head = j
            if token_span[-1] == end:
                tail = j
        if head == None:
            for j, token_span in enumerate(token_spans):
                if token_span[0] - start in [-1, -2]:
                    head = j
        if tail == None:
            for j, token_span in enumerate(token_spans):
                if token_span[1] - end in [1, 2]:
                    tail = j
        if head == None:
            for j, token_span in enumerate(token_spans):
                if token_span[0] - start in [-3, -4]:
                    head = j
        if tail == None:
            for j, token_span in enumerate(token_spans):
                if token_span[1] - end in [3, 4]:
                    tail = j
        if head == None:
            for j, token_span in enumerate(token_spans):
                if token_span[0] - start in [-5, -6]:
                    head = j
        if tail == None:
            for j, token_span in enumerate(token_spans):
                if token_span[1] - end in [5, 6]:
                    tail = j
        span = [head,tail]
        for k in range(head,tail+1):
            notes[k] = label
        spans.append(span)
    return spans,notes


def gen_trigger_data(event_data,tokenizer):
    sent = event_data['sent']['text']
    tokens = tokenizer.tokenize(sent)
    events = event_data['events']
    label_spans = []
    labels = []
    event_types = []
    event_spans = []
    trigger_texts = []
    for i,event in enumerate(events):
        event_type = event['event_type']
        event_span = match_span(tokens, sent, event['pos-in-sent'])
        label_span = event['trigger']['pos-in-sent']
        trigger_text = event['trigger']['text']
        trigger_texts.append(trigger_text)
        label_spans.append(label_span)
        labels.append(f"T{i}")
        event_spans.append(event_span)
        event_types.append(event_type)
    spans,notes = annotate(tokens, sent, label_spans, labels)
    tokens_in_span = []
    if_questioned = []
    blank_pattern = re.compile(r"[\t\n ]")
    for span in spans:
        tokens_in_span.append("".join(x.replace("##","") for x in tokens[span[0]:span[-1]+1]))
    for i,(true_trigger, annotated_trigger) in enumerate(zip(trigger_texts, tokens_in_span)):
        true_trigger = blank_pattern.sub("",true_trigger)
        annotated_trigger = blank_pattern.sub("",annotated_trigger)
        if true_trigger == annotated_trigger:
            if_questioned.append("o")
        else:
            if_questioned.append("wrong")
    return tokens, notes, spans, event_types, event_spans, trigger_texts, tokens_in_span, if_questioned

def get_token_span(text, label_span, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_spans = matcher(tokens, text)
    head = None
    tail = None
    start = label_span[0]
    end = label_span[-1]+1
    for j, token_span in enumerate(token_spans):
        if token_span[0] == start:
            head = j
        if token_span[-1] == end:
            tail = j
    if head == None:
        for j, token_span in enumerate(token_spans):
            if token_span[0] - start == -1:
                head = j
    if tail == None:
        for j, token_span in enumerate(token_spans):
            if token_span[1] - end == 1:
                tail = j
    span = [head,tail]
    return span



def matcher_old(tokens, text):
    pattern_a = re.compile("[^\t\n\xad ]")
    pattern_b = re.compile("^##")
    spans = []
    text_remain = text
    new_start = 0
    pattern_tashkeel = re.compile("[ًٌٍَُِّْ]")
    for i,token in enumerate(tokens):
        special = 0
        token = pattern_b.sub("", token)
        step = len(token)
        search_space = pattern_a.search(text_remain)
        span1,_ = search_space.span()
        head = new_start + span1
        tail = head + len(token)
        if pattern_tashkeel.search(text[head:tail+2]):
            tail += len(pattern_tashkeel.findall(text[head:tail+2]))
            special += len(pattern_tashkeel.findall(text[head:tail+2]))
        new_start = tail
        text_remain = text_remain[span1+len(token)+special:]
        spans.append((head,tail))
    return spans

class ara_ace_master:
    def __init__(self,file_paths,tokenizer):
        self.file_paths = file_paths
        self.paths = []
        self.texts = []
        self.sents = []
        self.event_mentions = []
        self.tokenizer = tokenizer
        for path in tqdm(self.file_paths):
            parser = Parser(path)
            parser.get_head()
            text = parser.sgm_text
            sents = parser.sents_with_pos
            event_mentions = parser.event_mentions
            self.paths.append(path)
            self.texts.append(text)
            self.sents.append(sents)
            self.event_mentions.append(event_mentions)
    def extract_trigger_data(self):
        file = open("trigger_span_align.txt","w+")
        self.trigger_datas = []
        self.wrong_trigger = 0
        self.n_trigger = 0
        self.bug = 0
        self.bug_mention = []
        bar = tqdm(enumerate(self.event_mentions))
        for i,event_mention in bar:
            event_data = get_event_data(event_mention, self.sents[i])
            trigger_data = []
            for mention in event_data:
                if mention['events'] == []:
                    continue
                tokens, notes, trigger_spans, event_types, event_spans, trigger_texts, tokens_in_span, if_questioned = gen_trigger_data(mention,self.tokenizer)
                trigger_data.append({"tokens":tokens,
                                    "notes":notes,
                                    "trigger_spans":trigger_spans,
                                    "event_types":event_types,
                                    "event_spans":event_spans})
                for j, (true, ann, if_q) in enumerate(zip(trigger_texts, tokens_in_span, if_questioned)):
                    if if_q == "wrong":
                        file.write(f"{self.file_paths[i]}\t{j}\t{true}\t{ann}\t{if_q}\n")
                        self.wrong_trigger += 1
                    else:
                        self.n_trigger += 1
                bar.set_postfix(wrong_data = self.wrong_trigger)
            self.trigger_datas.append(trigger_data)

class pad_collate:
    def __init__(self,max_length):
        self.max_length = max_length
    def padding(self,batch):
        ids = [sample[0] for sample in batch]
        ids = pad_sequence(ids, batch_first=True)[:,:self.max_length]
        labels = [sample[-1] for sample in batch]
        labels = pad_sequence(labels, batch_first=True)[:,:self.max_length]
        masks = torch.zeros(size = ids.size())
        masks[ids > 0] = 1
        return ids, labels, masks
    def __call__(self,batch):
        return self.padding(batch)

def data_check(data):
    tokens = tokenizer.convert_ids_to_tokens(data[0])
    sent = " ".join(x for x in tokens)
    ids = data[1]
    trigger = []
    for i,(token,id) in enumerate(zip(tokens,ids)):
        if id == 1:
            head = tokens[i]
            trigger.append(head)
        if id == 2:
            trigger[-1] += " "+tokens[i]
    trigger_text = "|".join(x for x in trigger)
    print(f"sentence = {sent},triggers = {trigger_text}")

def show_length_distribution(dataset):
    lengths = []
    for data in dataset:
        length = len(data[0])
        lengths.append(length)
    pd.DataFrame(lengths).hist()