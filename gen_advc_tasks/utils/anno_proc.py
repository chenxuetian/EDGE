def normalize_box_given_width_height(bbox: list, *, width=1920, height=1080):
    '''
    Normalize (namely to [0, 1]) the bounding box coordinates given the width and height of the image.
    The bounding box is represented as a list of 4 numbers: [x1, y1, x2, y2],
    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the bounding box.
    The width and height are the dimensions of the image, with default values of 1920 and 1080 respectively.
    The function returns a new list of 4 numbers which are the normalized coordinates of the bounding box.
    '''
    bbox_norm = bbox.copy()
    bbox_norm[0] = round(bbox_norm[0] / width, 4)
    bbox_norm[1] = round(bbox_norm[1] / height, 4)
    bbox_norm[2] = round(bbox_norm[2] / width, 4)
    bbox_norm[3] = round(bbox_norm[3] / height, 4)
    return bbox_norm

def raw_anno_2_text(anno: dict, *, normalize=True, show_index=False, show_title_and_description=True):
    '''
    - normalize: whether to normalize the bounding box coordinates (image height and width should be included in anno['viewport'])
    - show_index: whether to show the index of the element, intended for "Conversation intention" task generation where indices are helpful
    - show_title_and_description: whether to include the title and description of the webpage to give model (GPT 4o, Claude 3.5 sonnet) more context
    '''
    res = ''
    # URL
    res += "URL: " + anno['url'] + '\n'
    if show_title_and_description:
        res += "Title: " + anno['title'] + '\n'
        res += "Description: " + anno['description'] + '\n'
    width, height = anno['viewport'][0], anno['viewport'][1]
    # elements
    # { types }, "text", ariaLabel|title, [x1, y1, x2, y2]
    for i in range(len(anno['elements'])):
        ele = anno['elements'][i]
        if show_index:
            res += str(i) + '. '
        res += '{ '
        for type in ele['types']:
            res += type + ' '
        res += '}'

        res += ', "'
        res += ele['text'] + '", '
        
        opt_text = ele.get("ariaLabel", '')
        if not opt_text:
            opt_text = ele.get("title", '')
        res += f'"{opt_text}", '

        # if 'ariaLabel' in ele:
        #     res += (ele['ariaLabel'] if ele['ariaLabel'] else '') + ', '
        # else:
        #     res += ', '
        
        if normalize:
            res += str(normalize_box_given_width_height(ele['bbox'], width=width, height=height))
        else:
            res += str(ele['bbox'])
        res += '\n'
    return res