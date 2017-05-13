import os
import sys
import itertools
sys.path.append('../common')
import spacy
import numpy as np
from spacy.en import English
import re
nlp = English()
def replace_tokenizer(nlp):
    old_tokenizer = nlp.tokenizer
    nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(string.split())

replace_tokenizer(nlp)

NOUNS=[u'NNP',u'NN', u'NNS', u'NNPS',u'CD',u'PRP',u'PRP$']
ADJECTIVES=[u'JJ',u'JJR',u'JJS']
VERBS=[u'VB', u'VBN', u'VBD', u'VBG', u'VBP']
S_WORDS=[u'IN',u'WDT',u'TO',u'DT']

from okr import *

class V2:
	def __init__(self, name,nodes,edges, argument_alignment,sentences):
		self.name=okr.name #object_name
		self.edges=edges #a dictionary of edge_id to edge 
		self.nodes=nodes #a dictionary of node_id to node
		self.argument_alignment=argument_alignment # a list of groups of edges aligned to same argument
		self.sentences=sentences #original sentences (same as in okr)

		
class Node:
	def __init__(self, id, name, mentions, label,entailment):
		self.mentions=mentions #dictionary of mention_id to mention of node
		self.id=id #id of node- currenty "E+original_entity_id" or "P+original_proposition_id"
		self.name=name #name of node, the same as the name of original entity/proposition
		self.label=label #a set of all the mention terms of this node
		self.entailment=entailment #entailment graph for node (see entailment_graph object in okr)

class NodeMention:
	def __init__(self, id, sentence_id, indices, term, parent):
        	self.id = id #id of mention - same as in okr (an int)
        	self.sentence_id = sentence_id#sentence in which the mention appear
        	self.indices = indices #indices of the mention
       		self.term = term #words in the mention
        	self.parent = parent #id of the node to which the node mention belong

class Edge:
	def __init__(self, id, start_node,end_node):
		self.id=id #edge id - currently in the format of "Node_id_start_node+_+Node_id_end_node"
		self.start_node=start_node#start node
		self.end_node=end_node#end node
		self.mentions={} #dictionary of mention id to mention
		self.label=[]# set of all terms of all mentions

class EdgeMention:
	def __init__(self, id, sentence_id, indices, terms,template, all_nodes,main_pred, embedded, is_explicit):
		self.id = id #edge mention id - currently in the form of "original_proposition_id+_+original_mention_id" 
        	self.sentence_id = sentence_id #sentence of mention
        	self.indices = indices #indices of mention
       		self.terms = terms #terms of edge mention
       		self.template = template #template of edge mention
        	self.all_nodes=all_nodes #all nodes wich appear in the template
		self.embedded=embedded #edge mentions of embedded predicates in template. dictionary of embedded predicate to edge mentions. 
		self.embedded_edges={} #edges belonging to edge mentions of embedded predicates in template. 
						#dictionary of embedded predicate to edges. 
		self.is_explicit=is_explicit #is edge created from explicit proposition (like in okr)
		self.main_pred=main_pred #the main predicate of the edge
		self.parents=[] #all edges in which the edge mention appear




def change_template_predicate(template, terms, p_id):

	if terms==None or terms=="":
		#print ("as is: "+template+" "+p_id)
		return template
	template=" "+template+" "
	terms=" "+terms.lower()+" "
	#print(template+"\n")
	#print(terms+"\n")
	if template.find(terms)==-1:
		print(p_id+": non-consecutive predicate:|"+terms+"| in the template: |"+template+"|")
		return template
		#TODO: handle
	if len([n for n in re.finditer(terms, template)])>1:
		print("terms found more than once"+terms+template)
	new_template=template.replace(terms," ["+str(p_id)+"] ")
	#print (new_template+"\n")
	return new_template
	
def change_template_arguments(template, arguments):
	#print(template)
	#print({a:b.parent_id for a,b in arguments.iteritems()})
	new_template=template
	for arg_id, arg in arguments.iteritems():
		arg_str="[a"+str(int(arg_id)+1)+"]"
		element = "E" if arg.mention_type==0 else "P"
		element_id="["+element+str(arg.parent_id)+"]"
		new_template=new_template.replace(arg_str,element_id)			
	#print (new_template)
	return new_template

def get_embedded_mentions(arguments):
	embedded={}
	for arg_id, arg in arguments.iteritems():
		if arg.mention_type==1: #embedded predicate
			element_id="P"+str(arg.parent_id)
			#print element_id
			embedded[element_id]=element_id+"_"+str(arg.parent_mention_id)
	#print embedded
	return embedded
		
def extract_nodes_from_templates(template):
	nodes=[]
	while template.find("[")>-1:
		start=template.find("[")
		end=template.find("]")
		node=template[start+1:end]
		nodes.append(node)
		template=template[end+1:]
	return nodes
		
	
input_file=sys.argv[1]
okr = load_graph_from_file(input_file)
#entities to nodes:
Entity_Nodes={}	
for e_id,e in okr.entities.iteritems():
	new_entity_id="E"+str(e_id)
	mentions={m_id:NodeMention(m_id,m.sentence_id,m.indices,m.terms, new_entity_id) for m_id,m in e.mentions.iteritems()}
	Entity_Nodes[new_entity_id]=Node(new_entity_id,e.name, mentions, e.terms,e.entailment_graph)

#predicates to nodes:
Proposition_Nodes={}
Edge_Mentions=[]
for p_id,p in okr.propositions.iteritems():
	new_p_id="P"+str(p_id)
	prop_mentions={m_num:[[num,pos.orth_,pos.tag_] for num,pos in enumerate(nlp(unicode(" ".join(okr.sentences[m.sentence_id])))) if num in m.indices] for m_num,m in p.mentions.iteritems() if m.is_explicit} 
	new_terms={m_num:" ".join([str(word[1]) for word in m if word[2] not in S_WORDS])for m_num,m in prop_mentions.iteritems()}
	new_indices={m_num:[word[0] for word in m if word[2] not in S_WORDS ]for m_num,m in prop_mentions.iteritems()}
	new_terms_all=set([m for m in new_terms.values()])
	new_mentions={m_num: NodeMention(m_num,m.sentence_id,new_indices[m_num],new_terms[m_num], new_p_id) for m_num,m   in p.mentions.iteritems() if m.is_explicit}
	if (not (len(new_terms_all)==1 and [n for n in new_terms_all][0]=="")) and len(new_terms_all)>0:#not empty predicate
		Proposition_Nodes[new_p_id]=Node(new_p_id,p.name, new_mentions, new_terms_all,p.entailment_graph)

#create new templates:
	new_indices_edge={m_num:[word[0] for word in m if word[2] in S_WORDS ]for m_num,m in prop_mentions.iteritems()}
	new_terms_edge={m_num:" ".join([str(word[1]) for word in m if word[2] in S_WORDS])for m_num,m in prop_mentions.iteritems()}
	
	#replace predicates with nodes:
	new_templates_w_args= {m_num:   change_template_predicate(m.template, new_terms.get(m_num,None), new_p_id) for m_num, m in p.mentions.iteritems()}
	#replace arguments:
	new_templates= {m_num:change_template_arguments(template, p.mentions[m_num].argument_mentions) for m_num, template in new_templates_w_args.iteritems()}

	#get mentions of embedded predicates:
	embedded= {m_num:get_embedded_mentions(m.argument_mentions) for  m_num,m   in p.mentions.iteritems()}

	#assign main predicates only to propositions with a non-stop word predicate:
	main_predicates={m_num:new_p_id if not new_terms.get(m_num,"")=="" else None for m_num,m in p.mentions.iteritems()}
#extract edge mentions:
	Edge_Mentions=Edge_Mentions+[EdgeMention(new_p_id+"_"+str(m_num) , m.sentence_id, new_indices_edge.get(m_num,-1), new_terms_edge.get(m_num,""),new_templates[m_num], extract_nodes_from_templates(new_templates[m_num]), main_predicates[m_num],embedded[m_num], m.is_explicit) for m_num,m in p.mentions.iteritems()]
Nodes={}
Nodes.update(Entity_Nodes)
Nodes.update(Proposition_Nodes)

#create edges for all pairs of nodes that are connected by edge:
Edges={}
for edge_mention in Edge_Mentions:
	if  edge_mention.main_pred==None:
	   pairs=list(itertools.combinations(edge_mention.all_nodes, 2))
	else:
	   pairs=[(edge_mention.main_pred,node)  for node in edge_mention.all_nodes if not edge_mention.main_pred==node]
	for pair in pairs:
			edge_id="_".join(pair)
			if edge_id not in Edges:
				Edges[edge_id]=Edge(edge_id,pair[0],pair[1])
			Edges[edge_id].mentions[edge_mention.id]=edge_mention
			edge_mention.parents.append(edge_id)
#set edge labels:
for edge in Edges.values():
	edge.label=set([mention.template for mention in edge.mentions.values()])

#turn embedded predicate mentions to edges:
Edge_Mentions_dict={m.id:m for m in Edge_Mentions}
for edge in Edges.values():
	for mention in edge.mentions.values():
		new_embedded={}
		for e in mention.embedded:
			new_embedded[e]=Edge_Mentions_dict[mention.embedded[e]].parents
		mention.embedded_edges=new_embedded

#Add argument alinment links:
Args={}
Argument_Alignment={}
for p_id,p in okr.propositions.iteritems():
	new_p_id="P"+str(p_id)
	Args[new_p_id]={}
	for m in p.mentions.values():
		 for a_id,a in m.argument_mentions.iteritems():
			element = "E" if a.mention_type==0 else "P"
			element_id=element+str(a.parent_id)
			if a_id not in Args[new_p_id]:
				Args[new_p_id][a_id]=set()
	 		Args[new_p_id][a_id].add(element_id)
	alignment=[[new_p_id+"_"+element for element in v] for k,v in Args[new_p_id].iteritems() if len(v)>1 ]
	if (len(alignment)>0):
		Argument_Alignment[new_p_id]=alignment

#create final V2 object:
v2=V2(okr.name,Nodes,Edges, Argument_Alignment, okr.sentences)
		
	
	
	

