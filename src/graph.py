import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_chunk(self, chunk_id, text):
        self.G.add_node(chunk_id, type="chunk", text=text)

    def add_rule(self, rule_id, text):
        self.G.add_node(rule_id, type="rule", text=text)

    def add_element(self, element_id, name):
        self.G.add_node(element_id, type="element", name=name)

    def add_section(self, section_id, title):
        self.G.add_node(section_id, type="section", title=title)

    def link_chunk_rule(self, chunk_id, rule_id):
        self.G.add_edge(chunk_id, rule_id, type="contains")

    def link_rule_element(self, rule_id, element_id):
        self.G.add_edge(rule_id, element_id, type="applies_to")

    def link_element_section(self, element_id, section_id):
        self.G.add_edge(element_id, section_id, type="in_section")

    def expand_from_chunks(self, chunk_ids):
        rules = set()
        elements = set()

        for cid in chunk_ids:
            for _, rule_id in self.G.out_edges(cid):
                rules.add(rule_id)

        for rule_id in rules:
            for _, element_id in self.G.out_edges(rule_id):
                elements.add(element_id)

        expanded_rules = set()
        for element_id in elements:
            for rule_id, _ in self.G.in_edges(element_id):
                expanded_rules.add(rule_id)

        return list(expanded_rules)

    def get_texts(self, node_ids):
        texts = []
        for nid in node_ids:
            data = self.G.nodes[nid]
            if data["type"] == "rule":
                texts.append(data["text"])
        return texts
