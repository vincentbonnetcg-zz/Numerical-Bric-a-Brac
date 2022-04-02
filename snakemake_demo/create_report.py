import json
import argparse
import random
import os
import base64

def get_json_infos(json_path):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)
        comments = random.choices(data['comments'], k=3)
        shape = data['shape'][0]
        return shape, comments
        
    return None

def get_image_str(filepath):
    # From https://stackoverflow.com/questions/50817518/hard-code-markdown-images
    image_read = open(filepath, 'rb').read()
    image_64_encode = base64.encodestring(image_read) 
    image_string = str(image_64_encode)
    image_string = image_string.replace("\\n", "")
    image_string = image_string.replace("b'", "")
    image_string = image_string.replace("'", "")
    image_string = '<p><img src="data:image/png;base64,' + image_string + '"></p>'
    return image_string

def create_report(input_duration_pngs, input_time_pngs, input_jsons, output_file):
    with open(output_file, 'w') as file:
        file.write('# UFO Report\r\n')
        for i in range(len(input_duration_pngs)):
            shape, comments = get_json_infos(input_jsons[i])
            duration_img_str = get_image_str(input_duration_pngs[i])
            time_img_str = get_image_str(input_time_pngs[i])
            file.write('## {}\r\n'.format(shape))       
            file.write("<table><tr><th>{}</th><th>{}</th></tr></table>\r\n\r\n".format(duration_img_str, time_img_str))
            for comment in comments:
                file.write('+ {}\r\n'.format(comment))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_duration_pngs", nargs='+')
    parser.add_argument("-input_time_pngs", nargs='+')
    parser.add_argument("-input_jsons", nargs='+')
    parser.add_argument("-output_file")
    args = parser.parse_args() 

    create_report(args.input_duration_pngs,
                  args.input_time_pngs,
                  args.input_jsons,
                  args.output_file)

