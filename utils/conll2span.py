import numpy as np


def conll2span(label):

        """
        Input: array(['O', 'O', 'O', 'B-time', 'I-time', 'E-time', 'O', ...,], dtype='<U14')
        output: [(3, 6, 'time'), (...), (...)]
        """

        stack = []
        label = np.array(label)
        state = {}
        labels = []
        
        for idx, tag in enumerate(label):
            
            # Update state
            state['Nb'] = tag.split('-')[0]
            state['Nt'] = tag.split('-')[-1]
            
            # Single tag
            if state['Nb'] == "S":
                labels.append((idx,idx+1,state['Nt']))
                
            # Start tag
            elif state['Nb'] == "B": 
                stack.append((idx, state['Nt']))
                
            # end tag  
            elif state['Nb'] == "E":
                if state['Nt'] == stack[-1][1]:
                    temp_tag = stack.pop()
                    labels.append((temp_tag[0], idx+1, state['Nt']))
                else:
                    print("Error :: Unbalanced") 
                    breakpoint()

        if len(stack) != 0: 
            print("tag", tag)
            print("\nstack", stack)
            print('\nLabel', labels)
            raise "Error :: Unbalanced"

        return labels