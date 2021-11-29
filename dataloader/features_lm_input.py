import os
from tabulate import tabulate
from transformers import AutoTokenizer, AutoConfig

class InputLM():
    def __init__(self, lm_path, max_length) -> None:
        
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            lm_path, model_max_length=max_length)

    def __call__(self, tokens, entities):
        
        tokens = tokens.copy()
        entities = entities.copy()
        
        input_ids, attention_mask, encode_dict = self._tokenizer_input(
            tokens
        )
        
        shifted_entities = self._shifted_entities_index(
            input_ids, 
            entities, 
            encode_dict
        )
        
        lm_tokens=[self.tokenizer.decode(w)for w in input_ids]
        
        item = {}
        item['attention_mask'] = attention_mask
        item['input_ids'] = input_ids
        item['lm_tokens'] = lm_tokens
        item['lm_entities'] = shifted_entities
        item['encode_dict'] = encode_dict
        return item


    def _tokenizer_input(self, tokens):
        
        max_length = self.max_length
        start_id = self.tokenizer.bos_token_id
        end_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        
        encode_dict = {}
        input_ids = [start_id]
        
        for index in range(len(tokens)):

            word = tokens[index]
            shifted = len(input_ids)

            ids = self.tokenizer.encode(word)
            ids = ids[1:-1]

            input_ids.extend(ids)
            encode_dict[index]=(shifted, shifted+len(ids))

        # Add end of word
        input_ids.append(end_id)

        # Create mask
        num_ids = len(input_ids)
        mask = [1]*num_ids
        mask+= [0]*(max_length-num_ids)
        assert len(mask)==max_length, 'Error create mask'
        
        # Add padding
        input_ids+=[pad_id] * (max_length-num_ids)
        return input_ids, mask, encode_dict

    def _shifted_entities_index(self, input_ids, entities, encode_dict):
        
        shifted_entities = []
        
        # Shift labels index
        for index in range(len(entities)):
            entity = entities[index]
            entity_type = entity['entity_type']
            start, end = entity['span']
            text = entity['text']

            # shifting start, end
            (shifted_start, _) = encode_dict.get(start)
            (_, shifted_end) = encode_dict.get(end-1)

            decode_text = input_ids[shifted_start:shifted_end]
            decode_text = [self.tokenizer.decode(w) for w in decode_text]
            decode_text = "".join(decode_text)
            
            shifted_entities.append({
                'entity_type':entity_type,
                'span':[shifted_start, shifted_end],
                'text': decode_text
            })
            
        return shifted_entities

    @staticmethod
    def check_entities(sample):
        temp = [['original_en', 'orginal_span', 'decode_en', 'decode_span']]
        
        for index in range(len(sample['org_entities'])):
            
            org_entity = sample['org_entities'][index]
            original_ne = org_entity['text']
            original_span = org_entity['span']
            
            decode_entity = sample['entities'][index]
            decode_ne =decode_entity['text']
            decode_span = decode_entity['span']
            
            temp.append([original_ne, 
                        original_span, 
                        decode_ne, 
                        decode_span])
        print(tabulate(temp))

    @staticmethod
    def check_input_ids_and_mask(sample):
        temp = [['index', 'input_text', 'input_ids', 'mask']]
        for index in range(len(sample['input_ids'])):
            
            original_ids = sample['input_ids'][index]
            mask = sample['mask'][index]
            input_text = sample['input_text'][index]
            
            temp.append([index, input_text, original_ids, mask])
        print(tabulate(temp))


if __name__ == "__main__":
    tokens = [
        'การ', 'พบปะ', 'กับ', 'บรรดา', 'สื่อมวลชน', 'ใน', 'ครั้งนี้', 'ถือว่า', 
        'ปรับ', 'กระบวนการ', 'เชิงรุก', 'ของ', 'รัฐบาล', 'ภายใต้', 'การนำ', 
        'ของ', '_', 'พล', '.', 'อ.', 'สุรยุทธ์', '_', 'อีกครั้ง', '_', 'หลังจากที่', 
        'ได้', 'เปลี่ยนแปลง', 'การทำงาน', 'เพื่อ', 'ลด', 'แรงกดดัน', 'จาก', 'สังคม', 
        'มากขึ้น', '_', 'ที่มา', '_', '_', 'ผู้จัดการ', 'ออนไลน์']

    entities = [
        {'span': [17, 21], 'entity_type': 'person', 'text': 'พล . อ. สุรยุทธ์'},
        {'span': [17, 20], 'entity_type': 'title', 'text': 'พล . อ.'},
        {'span': [20, 21], 'entity_type': 'firstname', 'text': 'สุรยุทธ์'},
        {'span': [38, 40], 'entity_type': 'media', 'text': 'ผู้จัดการ ออนไลน์'},
        {'span': [38, 39], 'entity_type': 'media', 'text': 'ผู้จัดการ'}]

    # Preprocessing input lm
    input_lm = InputLM(
        lm_path="../resources/lm",
        max_length=80
    )

    lm_input = input_lm(tokens, entities)
    print("\n\nKeys, ",lm_input.keys())

    # input_ids
    print("\ninput_ids")
    print(lm_input['input_ids'])

    # mask
    print("\nattention_mask")
    print(lm_input['attention_mask'])

    # lm_tokens
    print("\nlm_tokens")
    print(lm_input['lm_tokens'])

    # lm_tokens
    print("\nencode_dict")
    print(lm_input['encode_dict'])

    # lm_tokens
    print("\nlm_entities")
    print(lm_input['lm_entities'])

    # Check span output
    print("\nlm_tokens[33:40]")
    print(lm_input['lm_tokens'][33:40])

    breakpoint()