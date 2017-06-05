import okr
graph1=okr.load_graph_from_file("../../data/baseline/test/car_bomb.xml")
for id,proposition in graph1.propositions.iteritems():
     print("proposition: "+str(id))
     for m_id,mention in proposition.mentions.iteritems():
           print ("\tmention: "+str(m_id))
           print "\ttemplate: "+mention.template
           print "\targuments:"
           for a_id,arg in mention.argument_mentions.iteritems():
                                   print "\t\tA"+a_id+": "+arg.desc
				   mention_type= "Entity: "if arg.mention_type==0 else "Proposition: "
				   print "\t\t\t"+mention_type+str(arg.parent_id)+" mention: "+str(arg.parent_mention_id)
				   print "\t\t\t"+arg.parent_name
