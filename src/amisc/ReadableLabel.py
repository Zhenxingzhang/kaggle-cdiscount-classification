import os


class ReadableLabel(object):
    def __init__(self,
                 readable_label_txt):
        # if not readable_label_txt:
        self.uid_lookup_path = os.path.join(readable_label_txt)
        self.node_lookup = self.load()

    def load(self):
        with open(self.uid_lookup_path, 'r') as inf:
            imagenet_classes = eval(inf.read())
        return imagenet_classes

    def get_human_readable(self, id_):
        id_ = id_ - 1
        label = self.node_lookup[id_]
        return label