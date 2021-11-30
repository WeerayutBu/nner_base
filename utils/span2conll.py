import pdb
import numpy as np

class Span2conll():
    def __init__(self, visualize=True, max_depth=8, SEP='[SEP]'):
        self.max_depth = max_depth
        self.visualize = visualize
        self.SEP = '@'
    
    def __call__(self, tokens, labels):
        """
        inputs : 
            tokens is a list of word
            labels is a list of tuple(start, end, tag)
            
        output :
            a sentence in conll format
        """
        output = self.span2conll(tokens, labels)
        return output
    
    def index_in_span(self, idx, entity_list, mode='start'):
    
        # Check mode
        if   mode=='start': mode=0
        elif mode=='end':   mode=1
        else:
            raise "Check mode"

        ## get all idx entities
        idx_entity_list = [p[mode] for p in entity_list]

        ## Get index of entity list that start with idx
        idx_entities =  np.where(np.array(idx_entity_list) == idx)[0]

        # There is not entity in the list
        if len(idx_entities) == 0:
            return False

        ## Return list of entities that start with the idx
        return [ entity for idx, entity in enumerate(entity_list) 
                if idx in idx_entities]
    

    def span2conll(self, words, labels):
        max_token=self.max_depth
        result_conll  = []
        entity_queue  = []
        entity_counts = 0

        # Sorted labels
        labels = sorted(labels, key=lambda x:(x[0], -x[1]))

        # mask entities
        labels = [(e[0], e[1], f"{str(idx+1)}{self.SEP}{e[2]}") 
                          for idx, e in enumerate(labels)]

        # Re-format to conll
        for idx, word in enumerate(words):
            # Push : when idx in start span of label --> sorted entity queue
            # Show : every words
            # Pop  : when idx in end span of label

            ## Push entity in entities queue
            # Check new entities from current idx.
            start_entities = self.index_in_span(idx, labels)

            # There are new entities.
            if start_entities :

                # Push new entities
                entity_queue.extend(start_entities)

                # Sort entities in the queue by (min_start_idx, max_stop)
                # print(entity_queue)
                entity_queue = sorted(entity_queue, key=lambda x:(x[0], -x[1]))

            ## Pop entity out of entity equeue
            end_entities = self.index_in_span(idx, labels, 'end')
            # print('\n',end_entities)
            # print(entity_queue)

            # Pop the entities from entities queue
            if end_entities:
                entity_queue = [end_en for end_en in entity_queue 
                                if end_en not in end_entities]
            #print(entity_queue)

            # Keep result
            temp_result = [ x[-1] for x in entity_queue]

            # temp_result = [ x[-1].split(self.SEP)[-1] 
            #                    for x in entity_queue]

            temp_result+=['O']*(max_token-len(temp_result))

            result_conll.append([word]+temp_result)
    #         result_conll.append(temp_result)

            ## Show ###
            if self.visualize:
                # print word
                token = word[0:max_token] \
                        if len(word) <= max_token \
                        else word[0:max_token]+'...' 

                print(f"\n{idx:<3} {token:<15} \t", end=' ')

                # print label
                for label in entity_queue:
                    label = label[-1]
                    label = label[0:max_token+3] \
                            if len(label) <= max_token \
                            else label[0:max_token]+'...' 

                    print(f"{label:<15}", end=' ')


        # Remove entities that stop at the last index
        entity_queue = [e for e in entity_queue if e[1]!=idx+1]
        if len(entity_queue) != 0:
            pdb.set_trace()

        return result_conll

    @staticmethod
    def processed_entities(label):
        temp_label = []
        for index in range(len(label)):
            _start, _end = label[index]['span']
            _tag = label[index]['entity_type']
            temp = (_start, _end, _tag)
            temp_label.append(temp)
        return temp_label
    
    @staticmethod
    def tag_bio(entities):
        num_tokens = len(entities)
        results = []
        for index in range(num_tokens):
            tag = entities[index]
            if tag=="O": 
                results.append(tag)
            elif tag!=entities[index-1]:
                tag = tag.split("@")[-1]
                results.append(f"B-{tag}")
            elif tag!="O":
                tag = tag.split("@")[-1]
                results.append(f"I-{tag}")
            else:
                raise "Error"
        return results

if __name__ == '__main__':
    tokens = ['ความคืบหน้า', 'หลัง', 'ศาลปกครอง', 'กลาง', 'มี', 'คำสั่ง', 'ไม่',
          'รับ', 'คำฟ้อง', 'และ', 'ไม่', 'คุ้มครอง', 'ชั่วคราว', 'ใน', 'คดี', 
          'ที่', 'ผู้ตรวจการแผ่นดิน', 'ยื่นฟ้อง', 'ต่อ', 'ศาลปกครอง', 'ว่า', '_', 
          'สำนักงาน', 'คณะกรรมการ', 'กิจการ', 'กระจายเสียง', '_', 'กิจการ', 
          'โทรทัศน์', 'และ', 'กิจการ', 'โทรคมนาคม', 'แห่งชาติ']

    labels = [(2, 4, 'org:other'), (23, 33, 'goverment'), (23, 33, 'goverment')]
    span2conll = Span2conll(visualize=True, max_depth=5)
    conll = span2conll(tokens, labels)
    conll
    
    # How to use get_bio
    get_bio(
        ['_', 'ราคา', 'อยู่', 'ที่', 'แก้ว', 'ละ', '_', '3050', '_', 'บาท'], 
        ['O', 'O', 'O', 'O', 'O', 'O', 'O', '18@money', '18@money','18@money']
    )