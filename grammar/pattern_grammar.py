#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import nltk

syntactic_compiled_grammar = {}
source_target_compiled_grammar = {}


class PatternGrammar:
    @property
    def syntactic_grammars(self):
        grammar = {
            1: """
                VBG_RB_DESRIBING_NN:    {   <NN|NN.><MD>?<RB|RB.>*<VB|VB.><RB|RB.>*(<CC|,>?<RB|RB.>?<VB|VB.>)+<NN|NN.>+ }""",
            2: """
                # the place was amazing |
                VBG_DESRIBING_NN: {<NN|NN.><VB|VB.>+<RB|RB.>*<VB|VB.>}""",
            3: """            
                # I loved the ambiance and the food. | enjoyed the taste.
                VBG_DESCRIBING_NN_V3 : {<VB|VB.>+<DT>?<IN>?<NN.|NN>+(<CC|,>?<DT>?<JJ|JJ.>*<NN|NN.>+)*}  # noqa nopep8""",
            4: """             
                # Amazingly satisfying food
                VBG_DESRIBING_NN_V2: {<RB|RB.>*<VB.|VBG>+<NN|NN.>}""",
            5: """
                VBG_DESRIBING_NN_V5 :{<JJ|JJ.><N.|NN.><VB.|VB><RB>*<VB.>}""",
            6: """            
                # perfect place to have varied options in burgers
                VBG_NN_DESCRIBING_NN: { <VBN><NN|NN.><IN><NN|NN.> }""",
            7: """            
                # improved on their service
                VBN_IN_PRP_NN: { <VBN><IN><PRP\$><NN> }""",
            8: """
                VBG_NN_DESCRIBING_NN: { <VBN><NN|NN.><IN><NN|NN.> }""",
            9: """
                VBN_DESCRING_THE_FOLLOWING_NOUN : { <RB|JJ.|JJ>*<VB|VB.>+<IN|DT>?(<CC|,|TO>?<DT>?<NN|NN.>+)+}""",
            10: """            
                #i love east village pizza
                VB_DESCRBING_NN : { <VB|VB.>(<CC|,>?<RB|RB.><JJ|JJ.>*<NN|NN.>*)+}""",

            11: """            
                # the place was ok and good
                # The Phirni here is a rich flavoured dessert | Place not worth visiting | the place was ok and good | the fish , chicken and biryani were so tastefull   # noqa nopep8
                JJ_DESCRIBING_NN_V4 :{(<CC|,>*<DT>?<JJ|JJ.>*<NN|NN.>+)+(<IN><NN.|N.>)*<RB|RB.>*<VB|VB.>+<IN>?<DT>?(<CC|,>*<RB|RB.>*<JJ|JJ.>+<NN|NN.>*)+} # noqa nopep8""",

            12: """            
                # this place is always crowded, noisy and full #JJ_VBG_RB_DESRIBING_NN
                # great food , service and ambience. | great food and service. | the served the lovely food. | they have a speedy delivery
                # food is amazing
                NN_IS_VBG : { <NN><VBZ><VB|VB.> }""",
            13: """            
                # they have a speedy delivery
                PRP_VB_NN : { <PRP><VBP|VB><DT>?<NN|NN.>+ }""",
            14: """
                # they have a speedy delivery
                PRP_VB_NN : { <PRP><VBP|VB><DT>?<NN|NN.>+ }""",
            15: """            
                # nice for trying some authentic chinese
                NN_VB_DT_JJ_NN: { <NN><IN><VBG><DT><JJ><NN|NN.> }
            """,
            16: """
                # vegans like me can also enjoy
                NN_MD_VB : {<NN|NN.><IN|PRP>*<MD><RB>?<VB><JJ>?} """,
            17: """
                # They have awesome Indian and Chinese!
                PR_VB_JJ_JJ : { <PRP><VBP><JJ>+<CC|JJ>* } """,
            18: """
                # impossible to order
                JJ_TO_NN_VB : { <JJ>(<TO><NN|VB>+)+ } """,
            19: """
                RB_BEFORE_NN: {<RB|RB.>+<JJ|JJ.>*<NN|NN.>+} """,
            20: """
                # not a fan of biryani and rolls
                NN_IN_DT_NN_reverse : { <NN|NN.>+<IN><DT>(<CC|,>?<DT>?<JJ|JJ.>*<NN|NN.>+)+ } """,
            21: """
                # place isnt worth the hype
                NN_IN_DT_NN: { <NN>+<IN><DT><NN> } """,
            22: """            # I was disappointed with the chicken tikka
                I_JJ_NN : { <JJ.|J.><IN><DT>?<NN.|N.>+} """,
            23: """
                # impeccable service
                NN_JJ : { <CC|DT><NN|NN.>+<JJ|JJ.>}""",
            24: """JJ_BEFORE_NN : {<RB|RB.>*<JJ|JJ.><NN|NN.>+}""",
            25: """
                # The asparagus, truffle oil, parmesan bruschetta is a winner.
                NN_desc_NN : { (<CC|,>*<DT>?<JJ|JJ.>*<N.|NN.>+)+<VB.|V.><IN>?<DT>?<N.|NN.>+} """,
            26: """
                # Avoid this place
                NN_DT_NN : { <NN.|N.><DT><NN.|N.>+}
                """,
            27: """
                # my favourite is the chicken biryani
                NN_desc_NN_reverse : { <NN.|N.><VB.|V.><DT><N.|NN.>+} """,
            28: """
                # grasps all NN if none rule captures the sentence
                NN_Phrase : { <JJ|JJ.>?<VB.|V.>?<FW|NN.|N.>+ }""",
            29: """                
                JJ_VBG_RB_DESRIBING_NN_2: {  ( <CC|,|TO>? <DT>? <JJ|JJ.>*<NN|NN.>+ )+ <WDT> <..|...>* <VB|VB.|JJ|JJ.|RB>+ }
                """,
            30: """
                JJ_VBG_RB_DESRIBING_NN: {   (<CC|,>?<JJ|JJ.>*<VB.|V.>?<NN|NN.>)+<RB|RB.>*<MD>?<WDT|DT>?<VB|VB.>?<RB|RB.>*(<CC|,>?<RB|RB.>?<VB|VB.|JJ.|JJ|RB|RB.>+)+}
                """

        }
        return grammar

    @property
    def source_target_extraction_grammars(self):
        SRC_TARGET_GRAMMAR = {
            'JJ_grammar': """JJ: {<RB|R.>*<JJ|JJ.>}""",
            'NN_JJ_desc': """"NN_JJ :  {<JJ|JJ.>+<NN|NN.>*}""",
            'NN_desc': """"RB_VB :  {<RB|RB.>*<VB.|VBG>+}""",
            # NN_MD_VB,
            'VB_desc': """VB: { <VB|VB.><JJ>? }""",
            'JJ_multi': """JJ: { <JJ|JJ.|>+(<,|CC>?<JJ>)* }""",
            # JJ_IN_NN,
            'JJ_IN_NN': """JJ: { <JJ|JJ.><NN>+<IN> }""",
            'JJ_any_IN': """JJ: { <JJ|JJ.>+<.*>*<IN> }""",
            'JJ_NN_end': """NN: { (<VB|VB.>?<JJ|JJ.>*<NN|NN.>+)$ }""",
            'NN_beg': """ NP: { (^)(<CC|,>*<DT>?<JJ|JJ.>*<VB|VB.>?<N.|NN.>+)+ } """,
            'NN_only': """NP: {<NN|NN.>+}""",
            'NN_IN': """ NN: {<VB|VB.>?<NN|NN.>+<IN>} """,
            'DT_NN': """ NN: {<DT>(<CC|,>*<DT>?<JJ|JJ.>*<VB|VB.>?<N.|NN.>+)+} """,
            'TO_NN': """NP: { (<TO><NN|NN.>+)+ }""",
            # JJ_IN_NN | NN_MD_VB,
            'NE_grammar': """NP: {<JJ>*<N.|NN.>+},
                     NP: {<NN.|N.>+}""",
            'NN_all': """NP: {(<FW|NN|NN.>+<JJ|JJ.|VB|VB.>?)*<JJ|JJ.|VB|VB.>*<NN|NN.>+}""",
            'NN_CC_JJ_multi': """NP:     { (<,|CC>*<JJ|JJ.>?<NN|NN.>+)+ }""",
            'NP_before_VB': """NP: {(<CC|,>*<DT>?<JJ|JJ.>*<VB|VB.>?<NN|NN.>+)+(<IN><N.|NN.>)*<RB|RB.>*<VB|VB.>+}""",
            # noqa nopep8
            'NP_After_VB_must': """NP : {<VB.|V.><DT><JJ|JJ.>*<N.|NN.>+}""",
            'NP_After_VB': """NP : {(<,|CC>*<JJ|JJ.>?<NN|NN.>+)*}""",
            # was of great taste.,
            'NP_After_VB_i': """NP : {(<,|CC>*<JJ|JJ.|VB|VB.>?<NN|NN.>+)*}""",
            # The food Hummus n Pita and Fish n Chips were 'mouth watering`.,
            'VB_JJ_RB_desc': """NN_JJ : {<JJ|JJ.|RB|RB.>*<VB|VB.>+}""",  # was amazing
            'RB_AFTER_VB': """VB_RB : {<VB|VB.><RB|RB.>+<JJ|JJ.|NN|NN.|VB|VB.>*}""",
            # the service was fast.  | fast -> RB,
            'VB_all': """VB_ : { <VB|VB.>+}""",

            # all verbs : i 'loved' the ambience.
            'RB_all': """RB : {<JJ|JJ.>*<RB|RB.>+}""",

            'JJ_all': """JJ : {<JJ|JJ.>+}""",
            'NN_FW_only': """NN_FW : { <VB|VB.>?<NN.|N.|FW>+ }""",
            'JJ_AFTER_VB': """VBG_JJ : { <VB.|VB><JJ|JJ.|NN|NN.>+}""",
            'JJ_NN_RB_VB':
                """ JJ : {<JJ|JJ.>+}
                    RB_JJ : {<RB|RB.>+<JJ|JJ.>*}
                    RB_VB : {<VB.|VB>+<RB|RB.>+}
                    RB : {<RB|RB.>+}
                    JJ : {<JJ|JJ.>+<NN|NN.>*}
                    VB : {<VB|VB.>+}
                    NN : { <NN|NN.>+}
                    """
        }
        return SRC_TARGET_GRAMMAR

    @staticmethod
    def extractor_mapping_dict():
        extractor_dict = {
            'JJ_DESCRIBING_NN_V4': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                    'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBG_RB_DESRIBING_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                    'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBN_DESCRING_THE_FOLLOWING_NOUN': {
                'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'JJ_VBG_RB_DESRIBING_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                       'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'JJ_VBG_RB_DESRIBING_NN_2': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                         'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBG_DESCRIBING_NN_V3': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                     'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBG_NN_DESCRIBING_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                     'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBG_DESRIBING_NN_V2': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                    'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBG_DESCRIBIN_NN_V4': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                    'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VB_DESCRBING_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'RB_BEFORE_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                             'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'JJ_BEFORE_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                             'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_JJ': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                      'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'JJ_IN_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                         'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'JJ_TO_NN_VB': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                            'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_MD_VB': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                         'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VBN_IN_PRP_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                              'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'VB_PRP_NNS': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                           'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'PR_VB_JJ_JJ': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                            'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'I_JJ_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                        'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_IN_DT_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                            'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_VB_DT_JJ_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                               'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_Phrase': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                          'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_desc_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                           'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_DT_NN': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                         'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_desc_NN_reverse': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                   'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')},
            'NN_IN_DT_NN_reverse': {'source': PatternGrammar().get_source_target_compiled_grammar('NP_before_VB'),
                                    'target': PatternGrammar().get_source_target_compiled_grammar('JJ_AFTER_VB')}
        }
        return extractor_dict


    def get_source_target_compiled_grammar(self, clause):
        global source_target_compiled_grammar
        compiled_grammar = source_target_compiled_grammar.get(clause, None)
        if compiled_grammar is None:
            compiled_grammar = self.compile_source_target_grammar(clause)
            source_target_compiled_grammar[clause] = compiled_grammar
        return compiled_grammar

    def compile_source_target_grammar(self, clause):
        return nltk.RegexpParser(self.source_target_extraction_grammars[clause])

    def compile_all_source_target_grammar(self):
        global source_target_compiled_grammar
        clauses = list(self.source_target_extraction_grammars.keys())
        for clause in clauses:
            _ = self.get_source_target_compiled_grammar(clause)
        return source_target_compiled_grammar

    def get_syntactic_grammar(self, index):
        global syntactic_compiled_grammar
        compiled_grammar = syntactic_compiled_grammar.get(index, None)
        if compiled_grammar is None:
            compiled_grammar = self.compile_syntactic_grammar(index)
            syntactic_compiled_grammar[index] = compiled_grammar
        return compiled_grammar

    def compile_syntactic_grammar(self, index):
        return nltk.RegexpParser(self.syntactic_grammars[index])

    def compile_all_syntactic_grammar(self):
        global syntactic_compiled_grammar
        indexes = list(self.syntactic_grammars.keys())
        for index in indexes:
            _ = self.get_syntactic_grammar(index)
        return syntactic_compiled_grammar
