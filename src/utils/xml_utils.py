import xml.etree.ElementTree as ET
from collections import Counter
def get_tags_from_root(element, current_depth=0, layers=None):
    if layers is None:
        layers = {}  # Initialize layers dictionary
    
    # Add current tag to the appropriate depth layer
    if current_depth not in layers:
        layers[current_depth] = []
    layers[current_depth].append(element.tag)
    
    # Recursively process children
    for child in element:
        get_tags_from_root(child, current_depth + 1, layers)
    
    return layers

def get_tags_from_xml_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    tags = get_tags_from_root(root)
    return tags

def get_counter_from_tags(tags):
    
    counter = {}
    for depth, tags in tags.items():
        cnt = Counter(tags)
        cnt_dict = {}
        for key, value in cnt.items():
            cnt_dict[key] = value
        counter[depth] = cnt_dict
    
    
    return counter

if __name__ == '__main__':
    
    tags = get_tags_from_xml_file('./test/test.xml')
    
    print("Tags in the XML file: ", tags)
    print("Length of tags: ", len(tags))
    
    for depth, tags in tags.items():
        print(f"Layer {depth}: {tags}")