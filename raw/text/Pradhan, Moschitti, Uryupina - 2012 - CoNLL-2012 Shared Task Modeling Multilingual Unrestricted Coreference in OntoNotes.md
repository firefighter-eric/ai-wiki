# Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes

- Source PDF: `raw/pdf/Pradhan, Moschitti, Uryupina - 2012 - CoNLL-2012 Shared Task Modeling Multilingual Unrestricted Coreference in OntoNotes.pdf`
- Source SHA256: `0721b6cc36570e82035b4b56a6764bff11377cf4a0a43b738e9c5902b92744fa`
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

CoNLL-2012 Shared Task:
      Modeling Multilingual Unrestricted Coreference in OntoNotes

      Sameer Pradhan          Alessandro Moschitti          Nianwen Xue
  Raytheon BBN Technologies,      University of Trento,          Brandeis University,
     Cambridge, MA 02138         38123 Povo (TN)           Waltham, MA 02453
         USA                              Italy                USA
         pradhan@bbn.com             moschitti@disi.unitn.it          xuen@cs.brandeis.edu

               Olga Uryupina                    Yuchen Zhang
                University of Trento,                    Brandeis University,
               38123 Povo (TN)                   Waltham, MA 02453
                            Italy                       USA
                   uryupina@gmail.com                        yuchenz@brandeis.edu

                Abstract                         Early work on corpus-based coreference resolu-
                                                         tion dates back to the mid-90s by McCarthy and
    The CoNLL-2012 shared task involved pre-        Lenhert (1995) where they experimented with deci-
      dicting coreference in English, Chinese, and                                                     sion trees and hand-written rules. Corpora to support     Arabic, using the ﬁnal version, v5.0, of the                                                    supervised learning of this task date back to the Mes-     OntoNotes corpus.  It was a follow-on to the
                                                  sage Understanding Conferences (MUC) (Hirschman     English-only task organized in 2011.  Un-
        til the creation of the OntoNotes corpus, re-       and Chinchor, 1997; Chinchor, 2001; Chinchor and
     sources in this sub-ﬁeld of language process-       Sundheim, 2003).  The de facto standard datasets
     ing were limited to noun phrase coreference,         for current coreference studies are the MUC and the
     often on a restricted set of entities, such as       ACE1 (Doddington et al., 2004) corpora. These cor-
     the ACE entities. OntoNotes provides a large-        pora were tagged with coreferring entities in the
      scale corpus of general anaphoric coreference                                             form of noun phrases in the text. The MUC corpora     not restricted to noun phrases or to a spec-                                                  cover all noun phrases in text but are relatively small     iﬁed set of entity types, and covers multi-
     ple languages. OntoNotes also provides ad-         in size. The ACE corpora, on the other hand, cover
      ditional layers of integrated annotation, cap-      much more data, but the annotation is restricted to a
      turing additional shallow semantic structure.        small subset of entities.
     This paper describes the OntoNotes annota-         Automatic identiﬁcation of coreferring entities
      tion (coreference and other layers) and then       and events in text has been an uphill battle for sev-
     describes the parameters of the shared task in-                                                           eral decades, partly because it is a problem that re-     cluding the format, pre-processing informa-                                                      quires world knowledge to solve and word knowl-      tion, evaluation criteria, and presents and dis-
     cusses the results achieved by the participat-       edge is hard to deﬁne, and partly owing to the lack
     ing systems.  The task of coreference has        of substantial annotated data. Aside from the fact
     had a complex evaluation history. Potentially         that resolving coreference in text is simply a very
    many evaluation conditions, have, in the past,        hard problem, there have been other hindrances that
    made it difﬁcult to judge the improvement in         further contributed to the slow progress in this area:
    new algorithms over previously reported re-
      sults.  Having a standard test set and stan-                                                                         (i) Smaller sized corpora such as MUC which cov-     dard evaluation parameters, all based on a re-                                                        ered coreference across all noun phrases. Cor-     source that provides multiple integrated anno-
      tation layers (syntactic parses, semantic roles,            pora such as ACE which are larger in size, but
    word senses, named entities and coreference)             cover a smaller set of entities; and
     and in multiple languages could support joint             (ii) low consistency in existing corpora annotated
     modeling and help ground and energize on-            with coreference — in terms of inter-annotator
     going research in the task of entity and event                                                    agreement (ITA) (Hirschman et al., 1998) —     coreference.                                                 owing to attempts at covering multiple coref-
                                                        erence phenomena that are not equally anno-
1  Introduction                                                                tatable with high agreement which likely less-
  The importance of coreference resolution for the                                                   ened the reliability of statistical evidence in the
entity/event detection task, namely identifying all                                                  form of lexical coverage and semantic related-
mentions of entities and events in text and clustering                                                        ness that could be derived from the data and
them into equivalence classes, has been well recog-
nized in the natural language processing community.        1http://projects.ldc.upenn.edu/ace/data/

                                         1


                 Proceedings of the Joint Conference on EMNLP and CoNLL: Shared Task, pages 1–40,
                    Jeju Island, Korea, July 13, 2012. c⃝2012 Association for Computational Linguistics
<a id="page-2"></a>

### PDF 第 2 页

used by a classiﬁer to generate better predic-   was to explore whether it could ﬁll this void and help
      tive models. The importance of a well-deﬁned   push the progress further — not only in coreference,
     tagging scheme and consistent ITA has been   but with the various layers of semantics that it tries
     well recognized and studied in the past (Poe-   to capture. As one of its layers,  it has created a
      sio, 2004; Poesio and Artstein, 2005; Passon-   corpus for general anaphoric coreference that cov-
     neau, 2004). There is a growing consensus that   ers entities and events not limited to noun phrases
      in order to take language understanding appli-   or a subset of entity types. The coreference layer
     cations such as question answering or distilla-   in OntoNotes constitutes just one part of a multi-
      tion to the next level, we need more consistent   layered, integrated annotation of shallow semantic
     annotation for larger amounts of broad cover-   structures in text with high inter-annotator agree-
     age data to train better automatic models for   ment. This addresses the ﬁrst issue.
      entity and event detection.                         In the language processing community, the ﬁeld
 (iii) Complex evaluation with multiple evaluation   of speech recognition probably has the longest his-
     metrics and  multiple  evaluation  scenarios,   tory of shared evaluations held primary by NIST3
     complicated with varying  training and  test    (Pallett, 2002).  In the past decade machine trans-
      partitions, led to situations where many re-   lation has been a topic of shared evaluations also
     searchers report results with only one or a few   by NIST4. There are many syntactic and semantic
     of the available metrics and under a subset of   processing tasks that are not quite amenable to such
     evaluation scenarios. This has made it hard to   continued evaluation efforts.  The CoNLL shared
     gauge the improvement in algorithms over the   tasks over the past 15 years have ﬁlled that gap, help-
     years (Stoyanov et al., 2009), or to determine   ing establish benchmarks and advance the state of
    which particular areas require further attention.   the art in various sub-ﬁelds within NLP. The impor-
    Looking at various numbers reported in litera-   tance of shared tasks is now in full display in the
      ture can greatly affect the perceived difﬁculty   domain of clinical NLP (Chapman et al., 2011) and
     of the task. It can seem to be a very hard prob-   recently a coreference task was organized as part
    lem (Soon et al., 2001) or one that is relatively   of the i2b2 workshop (Uzuner et al., 2012).  The
     easy (Culotta et al., 2007).                     computational learning community is also witness-
 (iv) the knowledge bottleneck which has been a   ing a shift towards joint inference based evaluations,
     well-accepted ceiling that has kept the progress   with the two previous CoNLL tasks (Surdeanu et al.,
      in this task at bay.                            2008; Hajiˇc et al., 2009) devoted to joint learning of
                                                        syntactic and semantic dependencies. A SemEval-
  These issues suggest  that the following steps   2010 coreference task (Recasens et al., 2010) was
might take the community in the right direction to-   the ﬁrst attempt to address the second issue.   It
wards improving the state of the art in coreference   included six different Indo-European languages —
resolution:                                           Catalan, Dutch, English, German, Italian, and Span-
                                                              ish. Among other corpora, a small subset (∼120K)
  (i) Create  a  large  corpus  with  high  inter-   of English portion of OntoNotes was used for this
     annotator agreement possibly by restricting   purpose.  However, the lack of a strong participa-
     the coreference annotating to phenomena that   tion prevented the organizers from reaching any ﬁrm
     can be annotated with high consistency, and   conclusions. The CoNLL-2011 shared task was an-
     covering an unrestricted set of entities and   other attempt to address the second issue. It was well
      events; and                                       received, but the shared task was only limited to the
  (ii) Create a standard evaluation scenario with an   English portion of OntoNotes. In addition, the coref-
     ofﬁcial evaluation setup, and possibly several   erence portion of OntoNotes did not have a concrete
     ablation settings to capture the range of perfor-   baseline prior to the 2011 evaluation, thereby mak-
     mance.  This can then be used as a standard   ing it challenging for participants to gauge the per-
    benchmark by the research community.         formance of their algorithms in the absence of es-
 (iii) Continue to improve learning algorithms that   tablished state of the art on this ﬂavor of annotation.
      better incorporate world knowledge and jointly   The closest comparison was to the results reported
     incorporate information from other layers of   by Pradhan et al. (2007b) on the newswire portion of
      syntactic and semantic annotation to improve   OntoNotes. Since the corpus also covers two other
     the state of the art.                            languages from completely different language fami-
                                                                   lies, Chinese and Arabic, it provided a great oppor-
  One  of  the many  goals  of  the  OntoNotes   tunity to have a follow-on task in 2012 covering all
project2 (Hovy et al., 2006; Weischedel et al., 2011)
                                                                          3http://www.itl.nist.gov/iad/mig/publications/ASRhistory/index.html
   2http://www.bbn.com/nlp/ontonotes                                    4http://www.itl.nist.gov/iad/mig/tests/mt/

                                         2
<a id="page-3"></a>

### PDF 第 3 页

three languages. As we will see later, peculiarities  2  The OntoNotes Corpus
of each of these languages had to be considered in     The OntoNotes project has created a large-scale
creating the evaluation framework.                  corpus of accurate and integrated annotation of mul-
                                                              tiple levels of the shallow semantic structure in text.
                                           The English and Chinese language portion com-  The ﬁrst systematic learning-based study in coref-
                                                         prises roughly one million words per language oference resolution was conducted on the MUC cor-
                                                 newswire, magazine articles, broadcast news, broad-pora, using a decision tree learner, by Soon et al.
                                                         cast conversations, web data and conversational(2001). Signiﬁcant improvements have been made
                                                 speech data. The English subcorpus also containsin the ﬁeld of language processing in general, and
                                               an additional 200K words of the English translationimproved learning techniques have pushed the state
                                                     of the New Testament as Pivot Text. The Arabic por-of the art in coreference resolution forward (Mor-
                                                         tion is smaller, comprising 300K words of newswireton, 2000; Harabagiu et al., 2001; McCallum and
                                                               articles. The hope is that this rich, integrated an-Wellner, 2004;  Culotta et  al., 2007; Denis and
                                                      notation covering many layers will allow for richer,Baldridge, 2007; Rahman and Ng, 2009; Haghighi
                                                       cross-layer models and enable signiﬁcantly betterand Klein, 2010).  Researchers have continued to
                                                  automatic semantic analysis.  In addition to coref-ﬁnd novel ways of exploiting ontologies such as
                                                      erence, this data is also tagged with syntactic trees,WordNet. Various knowledge sources from shallow
                                                      propositions for most verb and some noun instances,semantics to encyclopedic knowledge have been ex-
                                                             partial verb and noun word senses, and 18 named en-ploited (Ponzetto and Strube, 2005; Ponzetto and
                                                                  tity types. Manual annotation of a large corpus withStrube, 2006; Versley, 2007; Ng, 2007).  Given
                                                     multiple layers of syntax and semantic informationthat WordNet is a static ontology and as such has
                                                                  is a costly endeavor. Over the years in the devel-limitation on coverage, more recently, there have
                                            opment of this corpus, there were various prioritiesbeen successful attempts to utilize information from
                                                           that came into play, and therefore not all the data inmuch larger, collaboratively built resources such as
                                                      the corpus could be annotated with all the differentWikipedia (Ponzetto and Strube, 2006). More re-
                                                        layers of annotation. However, such multi-layer an-cently researchers have used graph based algorithms
                                                        notations, with complex, cross-layer dependencies,(Cai et al., 2011a) rather than pair-wise classiﬁca-
                                            demands a robust, efﬁcient, scalable storage mech-tions. For a detailed survey of the progress in this
                                              anism while providing efﬁcient, convenient, inte-ﬁeld, we refer the reader to a recent article (Ng,
                                                     grated access to the the underlying structure.  To2010) and a tutorial (Ponzetto and Poesio, 2009)
                                                             this effect, it uses a relational database representa-dedicated to this subject. In spite of all the progress,
                                                         tion that captures both the inter- and intra-layer de-current techniques  still rely primarily on surface
                                                  pendencies and also provides an object-oriented APIlevel features such as string match, proximity, and
                                                         for efﬁcient, multi-tiered access to this data (Prad-edit distance; syntactic features such as apposition;
                                              han et al., 2007a). This facilitates the extraction ofand shallow semantic features such as number, gen-
                                                       cross-layer features in integrated predictive modelsder, named entities, semantic class, Hobbs’ distance,
                                                           that will make use of these annotations.etc. Further research to reduce the knowledge gap is
                                                OntoNotes comprises the following layers of an-essential to take coreference resolution techniques to
                                                        notation:the next level.

                                              • Syntax — A layer of syntactic annotation for
  The rest of the paper is organized as follows: Sec-                                                          English, Chinese and Arabic based on a revised
tion 2 presents an overview of the OntoNotes cor-                                                           guidelines for the Penn Treebank (Marcus et
pus.  Section 3 describes the range of phenomena                                                                               al., 1993; Babko-Malaya et al., 2006), the Chi-
annotated in OntoNotes, and language-speciﬁc is-                                                      nese Treebank (Xue et al., 2005) and the Arabic
sues. Section 4 describes the shared task data and                                                    Treebank (Maamouri and Bies, 2004).
the evaluation parameters, with Section 4.4.2 exam-
                                              • Propositions — The proposition structure ofining the performance of the state-of-the-art tools
                                                        verbs based on revised guidelines for the En-on all/most intermediate layers of annotation. Sec-
                                                              glish PropBank (Palmer et al., 2005; Babko-tion 5 describes the participants in the task.  Sec-
                                                Malaya et al., 2006), the Chinese PropBanktion 6 brieﬂy compares the approaches taken by var-
                                                (Xue and Palmer, 2009) and the Arabic Prop-ious participating systems.  Section 7 presents the
                                              Bank (Palmer et al., 2008; Zaghouani et al.,system results with some analysis. Section 8 com-
                                                        2010).pares the performance of the systems on the a subset
of the Engish test set that corresponds with the test     • Word Sense — Coarse-grained word senses
set used for the CoNLL-2011 evaluation. Section 9        are tagged for the most frequent polysemous
draws some conclusions.                                 verbs and nouns, in order to maximize token

                                         3
<a id="page-4"></a>

### PDF 第 4 页

coverage. The word sense granularity is tai-   the semantic types of NP entities that can be consid-
     lored to achieve 90% inter-annotator agreement   ered for coreference, and in particular, coreference
     as demonstrated by Palmer et al. (2007). These    is not limited to ACE types. The guidelines are fairly
     senses are deﬁned in the sense inventory ﬁles.   language independent. We will look at some salient
     In case of English and Arabic languages, the   aspects of the coreference annotation in OntoNotes.
     sense-inventories (and frame ﬁles) are deﬁned   For more details, and examples, we refer the reader
     separately for each part of speech that is real-   to the release documentation. We will primarily use
     ized by the lemma in the text.  For Chinese,   English examples to describe various aspects of the
    however the sense inventories (and frame ﬁles)   annotation and use Chinese and Arabic examples es-
     are deﬁned per lemma — independent of the   pecially to illustrate phenomena not observed in En-
     part of speech realized in the text.  For the    glish, or that have some language speciﬁc peculiari-
     English portion of OntoNotes, each individual    ties.
     sense has been connected to multiple WordNet                                                     3.1  Noun Phrases
     senses. This provides users direct access to the                                             The mentions over which IDENT coreference ap-    WordNet semantic structure.  There is also a                                                           plies are typically pronominal, named, or deﬁnite    mapping from the OntoNotes word senses to                                                  nominal. The annotation process begins by automat-    PropBank frames and to VerbNet (Kipper et                                                            ically extracting all of the NP mentions from parse       al., 2000) and FrameNet (Fillmore et al., 2003).                                                           trees in the syntactic layer of OntoNotes annotation,     Unfortunately, owing to lack of comparable re-                                                though the annotators can also add additional men-     sources as comprehensive as WordNet in Chi-                                                         tions when appropriate. In the following two exam-     nese or Arabic, neither language has any inter-                                                       ples (and later ones), the phrases in bold form the     resource mappings available.                                                         links of an IDENT chain.
  • Named Entities — The corpus was tagged
     with a set of 18 well-deﬁned proper named en-    (1) She had a good suggestion and it was unani-
      tity types that have been tested extensively for       mously accepted by all.
     inter-annotator agreement by Weischedel and
                                                          (2) Elco Industries Inc. said it expects net income     Burnstein (2005).
                                                              in the year ending June 30, 1990, to fall below a  • Coreference — This layer captures general                                                           recent analyst’s estimate of $ 1.65 a share. The     anaphoric coreference that covers entities and                                                   Rockford, Ill. maker of fasteners also said it     events not limited to noun phrases or a lim-                                                         expects to post sales in the current ﬁscal year     ited set of entity types (Pradhan et al., 2007b).                                                                that are “slightly above” ﬁscal 1989 sales of $       It considers all pronouns (PRP, PRP$), noun                                                155 million.     phrases (NP) and heads of verb phrases (VP)
     as potential mentions. Unlike English, Chinese                                          Noun phrases (NPs) in Chinese can be complex
    and Arabic have dropped subjects and objects                                            noun phrases or bare nouns (nouns that lack a de-
    which were also considered during coreference                                                    terminer such as “the” or “this”). Complex noun
     annotation5. We will take a look at this in detail                                                    phrases contain structures modifying the head noun,
     in the next section.                                                     as in the following examples:
3  Coreference in OntoNotes                       (3) (他担任总统任内最后一次的(亚太经
  General anaphoric coreference that spans a rich    济合作会议(高峰会))).
set of entities and events — not restricted to a few       ((His last APEC (summit meeting)) as the
types, as has been characteristic of most coreference        President)data available until now — has been tagged with a
high degree of consistency in the OntoNotes corpus.    (4) (越南统一后(第一位前往当地访问的
Two different types of coreference are distinguished:    (美国总统)))
Identity (IDENT), and Appositive (APPOS). Identity       ((The ﬁrst (U.S. president)) who went to visitcoreference (IDENT) is used for anaphoric corefer-                                                  Vietnam after its uniﬁcation)ence, meaning links between pronominal, nominal,
and named mentions of speciﬁc referents. It does not                                                        In these examples, the smallest phrase in paren-include mentions of generic, underspeciﬁed, or ab-                                                      theses is the bare noun. The longer phrase in paran-stract entities. Appositives (APPOS) are treated sep-                                                      theses includes modifying structures. All the expres-arately because they function as attributions, as de-                                                      sions in the parantheses, however, share the samescribed further below. Coreference is annotated for                                               head noun,  i.e., “高峰会(summit meeting)”, andall speciﬁc entities and events. There is no limit on
                               “美国总统(U.S. president)” respectively. Nested
   5As we will see later these are not used during the task.      noun phrases, or nested NPs, are contained within

                                         4
<a id="page-5"></a>

### PDF 第 5 页

longer noun phrases. In the above example, “sum-     Pronouns from classical Chinese such as 其中
mit meeting” and “U.S. president” are nested NPs.   (among which), 其(he/she/it), 之(he/she/it) are also
Wherever NPs are nested, the largest logical span is   linked with other mentions to which they refer.
used in coreference.                                    In Arabic, the following pronouns are corefer-
                                               enced – nominative personal pronouns (subject) and3.2  Verbs
                                                   demonstrative pronouns which are detached. Sub-  Verbs are added as single-word spans if they can                                                             ject pronouns are often null in Arabic; overt subjectbe coreferenced with a noun phrase or with another                                                pronouns are rare, but do occur.
verb. The intent is to annotate the VP, but the single-           á K@ /  ÕæK@ /  AÒJK@ /  ám  ' / áë / Ñë / AÒë
word verb head is marked for convenience.  This                                                                                                         (We, you, they)
includes morphologically related nominalizations as                                                                   ù ë / ñë / I K@ /   AK@
in (5) and noun phrases that refer to the same event,                                                                                                                                                       (I, you, she, he)
even if they are lexically distinct from the verb as in
(6). In the following two examples, only the chains     Object pronouns are attached to the verb (direct
related to the growth event are shown in bold. The   objects) or preposition (indirect objects)                                                                                                                      Aë /  é / ¼ / øArabic translation of the same example identiﬁes
                                                                                                (Me, you, him, her)mentions using parantheses.
                                                            á ë / Ñë /  AÒ» / á » /  Õ» /   AK
 (5) The European economy grew rapidly over the                                                 (Us, you, them)
     past years, this growth helped raising ....           and, possessive (adjectival) pronouns are identical
     H@ñJË@ ÈCg  é«Qå .  ú     B@ XAJ¯B @ (   AÖß ) Y®Ë                                G.ðPð                          to object pronouns, but are attachedAë to/ nouns.é / ¼ / ø                                                           ... ©¯P ú ¯ ÑëA ( ñÒJË@   @ Yë )  , éJ AÖÏ@                                           (My, your, his, her)
                                                            á ë / Ñë /  AÒ» / á » /  Õ» /   AK
 (6) Japan’s domestic sales of cars, trucks and buses                                                (Our, your, their)
     in October rose 18% from a year earlier to                                                 Pronouns such as 你，您，你们，大家，各位
    500,004 units, a record for the month, the Japan                                                can be considered generic. In this case, they are not
    Automobile Dealers’ Association said.  The                                                     linked to other generic mentions in the same dis-
     strong growth followed year-to-year increases                                                      course. For example,
     of 21% in August and 12% in September.
                                                          (8) 请*大家带好自己的随身物品。*大家请
3.3  Pronouns                  下车。
  All pronouns and demonstratives are linked to        Please take your belongings with *you. Please
anything that they refer to, and pronouns in quoted        get off the train, *everyone.
speech are also marked. Expletive or pleonastic pro-
nouns (it, there) are not considered for tagging, and      In Chinese, if the subject or object can be recov-
generic you is not marked. In the following exam-   ered from the context, or if it is of little interest for
ple, the pronoun you and it would not be marked. (In   the reader/listener to know,  it can be omitted.  In
this and following examples, an asterisk (*) before a   the Chinese Treebank, a small *pro* in inserted in
boldface phrase identiﬁes entity/event mentions that   positions where the subject or object is omitted. A
would not be tagged in the coreference annotation.)   *pro* can be replaced by overt NPs if they refer to
                                                      the same entity or event, and the *pro* and its overt
 (7) Senate majority leader Bill Frist likes to tell   NP antecedent do not have to be in the same sen-
     a story from his days as a pioneering heart   tence. Exactly what *pro* stands for is determined
     surgeon back in Tennessee. A lot of times,   by the linguistic context in which it appears.
      Frist recalls, *you’d have a critical patient ly-
     ing there waiting for a new heart, and *you’d    (9) 吉林省主管经贸工作的副省长全哲洙说：“
    want to cut, but *you couldn’t start unless *you       (*pro*) 欢迎国际社会同(我们) 一道，共
    knew that the replacement heart would make   同推进图门江开发事业，促进区域经济发
      *it to the operating room.            展，造福东北亚人民。
                                             Quan  Zhezhu,  Vice  Governor  of  Jinlin
  In Chinese, all the following pronouns — 你，        Province who is in charge of economics and
我，他, 她, 它，你们，我们，他们，它们，         trade, said: “(*pro*) Welcome international
我, 您, 咱们(you, me, he, she, and so on), and         societies to join (us) in the development of Tu-
demonstrative pronouns — 这个，那个，这些, 那      men Jiang, so as to promote regional economic
些(this, that, these, those) in singular, plural or pos-       development and beneﬁt people in Northeast
sessive forms are linked to anything they refer to.           Asia.

                                         5
<a id="page-6"></a>

### PDF 第 6 页

Sometimes, *pro*s cannot be recovered in the   (15) ParentsX  should be  involved  with  theirX
text—i.e., an overt NP cannot be identiﬁed as their         children’s education at home, not in school.
antecedent in the same text — and therefore they are      TheyX should see to it that theirX kids don’t
not linked. For instance, the *pro* in existential sen-        play truant; theyX should make certain that
tences usually cannot be recovered or linked in the        the children spend enough time doing home-
annotation, as in the following example:                 work; theyX should scrutinize the report card.
                                                   ParentsY are too likely to blame schools for the
(10) (*pro*) 有二十三顶高新技术项目进区开        educational limitations of theirY children.  If
  发。                                            parentsZ are dissatisﬁed with a school, theyZ
    There are 23 high-tech projects under develop-        should have the option of switching to another.
    ment in the zone.
                                                     In (16) below, the verb “halve” cannot be linked to  Also, if *pro* does not refer to a speciﬁc entity or                                                  “a reduction of 50%”, since “a reduction” is indeﬁ-event, it is considered generic *pro* and not linked                                                               nite.as in (11).
(11) 肯德基、麦当劳等速食店全大陆都推   (16) Argentina said  it will ask creditor banks to
   出了(*pro*) 买套餐赠送布质或棉质圣       *halve its foreign debt of $64 billion — the
                                                             third-highest in the developing world  .  Ar-  诞老人玩具的促销.
                                                         gentina aspires to reach *a reduction of 50%     In Mainland China, fast food restaurants such
                                                              in the value of its external debt.     as Kentucky Fried Chicken and McDonald’s
    have launched their promotional packages by
                                                     3.5  Pre-modiﬁers     providing  free  cotton Santa  toys  for each
                                                    Proper pre-modiﬁers can be coreferenced, but    combo (*pro*) purchased.                                                   proper nouns that are in a morphologically adjectival
   Finally, *pro*s in idiomatic expressions are not   form are treated as adjectives, and are not corefer-
linked. Similar to Chinese, Arabic null subjects and   enced. For example, adjectival forms of GPEs such
objects are also eligible for coreference and treated   as Chinese in “the Chinese leader”, would not be
similarly. In the Arabic Treebank, these are marked   linked. Thus we could coreference United States in
with just an “*”. There exists few of these instances   “the United States policy” with another referent, but
in English — marked (yet differently) with a *PRO*   not American in “the American policy.” GPEs and
in the treebank and which are connected in Prop-   Nationality acronyms (e.g.  U.S.S.R. or U.S.).  are
Bank annotation but not in coreference.                also considered adjectival.  Pre-modiﬁer acronyms
                                                can be coreferenced unless they refer to a national-
3.4  Generic mentions                                        ity. Thus in the examples below, FBI can be corefer-
  Generic nominal mentions can be linked with re-                                               enced to other mentions, but U.S. cannot.
ferring pronouns and other deﬁnite mentions, but not
with other generic nominal mentions.                                                     (17) FBI spokesman
  This would allow linking of the bolded mentions
in (12) and (13), but not in (14).                      (18) *U.S. spokesman
(12) Ofﬁcials said they are tired of making the      In Chinese adjectival and nominal forms of GPEs
    same statements.                                 are not morphologically distinct, and in such cases
(13) Meetings are most productive when they are   the annotator decides whether it is an adjectival us-
     held in the morning. Those meetings, how-   age. Usually if something is tagged as NORP then it
     ever, generally have the worst attendance.          is not considered as a mention.
(14) Allergan Inc.   said  it received approval to     Dates and monetary amounts can be considered
      sell the PhacoFlex intraocular lens, the ﬁrst   part of a coreference chain even when they occur as
     foldable silicone lens available for *cataract   pre-modiﬁers.
     surgery. The lens’ foldability enables it to be
     inserted in smaller incisions than are now pos-   (19) The current account deﬁcit on France’s balance
     sible for *cataract surgery.                          of payments narrowed to 1.48 billion French
                                                           francs ($236.8 million) in August from a re-
  Bare plurals, as in (12) and (13), are always con-                                                         vised 2.1 billion francs in July, the Finance
sidered generic.  In example (15) below, there are                                                        Ministry said. Previously, the July ﬁgure was
three generic instances of parents. These are marked                                                         estimated at a deﬁcit of 613 million francs.
as distinct IDENT chains (with separate chains dis-
tinguished by subscripts X, Y and Z), each contain-   (20) The company’s $150 offer was unexpected.
ing a generic and the related referring pronouns.         The ﬁrm balked at the price.

                                         6
<a id="page-7"></a>

### PDF 第 7 页

3.6  Copular verbs                                 Deictic expressions such as now, then, today, tomor-
   Attributes signaled by copular structures are not   row, yesterday, etc. can be linked, as well as other
marked; these are attributes of the referent they mod-   temporal expressions that are relative to the time of
ify, and their relationship to that referent will be cap-   the writing of the article, and which may therefore
tured through word sense and proposition annota-   require knowledge of the time of the writing to re-
tion.                                                solve the coreference. Annotators were allowed to
                                                  use knowledge from outside the text in resolving
(21) JohnX  is a linguist.  PeopleY are nervous   these cases.  In the following example, the end of
    around JohnX, because heX always corrects    this period and that time can be coreferenced, as can
     theirY grammar.                                     this period and from three years to seven years.
                                                     (26) The limit could range from three years to
  Copular (or ’linking’) verbs are those verbs that       seven yearsX, depending on the composition
function as a copula and are followed by a subject        of the management team and the nature of its
complement. Some common copular verbs are: be,         strategic plan. At (the end of (this period)X)Y,
appear, feel, look, seem, remain, stay, become, end        the poison pill would be eliminated automati-
up, get. Subject complements following such verbs         cally, unless a new poison pill were approved
are considered attributes and are not linked. Since       by the then-current shareholders, who would
Called is copular, neither IDENT nor APPOS corefer-       have an opportunity to evaluate the corpora-
ence is marked in the following case.                         tion’s strategy and management team at that
                                                     timeY.
(22) Called Otto’s Original Oat Bran Beer, the brew
     costs about $12.75 a case.                          In multi-date temporal expressions, embedded
                                                      dates are not separately connected to other mentions
  Some examples of copular verbs in Chinese are   of that date.  For example in Nov.  2, 1999, Nov.
是(to be) and 为(to be, to serve as). In addition,   would not be linked to another instance of November
other verbs (particularly so-called light verbs) that    later in the text.
trigger an attributive reading on the following NP: 成   3.9  Appositives
为(become), (当)选为(is elected), 称为(is called),     Because they logically represent attributions, ap-
(好)像(looks like), 叫做(is called), etc.               positives are tagged separately from Identity coref-
                                                      erence. They consist of a head, or referent (a noun
(23) (上海)是*(中国最大的城市)。(上海)发   phrase that points to a speciﬁc object/concept in the
  展得很快。                                    world), and one or more attributes of that referent.
    (Shanghai) is *(the largest city in China).  An appositive construction contains a noun phrase
    (Shanghai) develops fast.                         that modiﬁes an immediately-adjacent noun phrase                                                      (separated only by a comma, colon, dash, or paren-
                                                              thesis).  It often serves to rename or further deﬁne
  In the above example, the two mentions of 上                                                      the ﬁrst mention. Marking appositive constructions
海(Shanghai) co-refer with each other, but the en-   allows capturing the attributed property even though
tity does not co-refer with 中国最大的城市(the   there is no explicit copula.
largest city in China).
                                                     (27) Johnhead, a linguistattribute3.7  Small clauses
  Like copulas, small clause constructions are not                                             The head of each appositive construction is distin-marked as coreferent.  The following example is                                                  guished from the attribute according to the followingtreated as if the copula were present (“John consid-                                                          heuristic speciﬁcity scale, in a decreasing order fromers Fred to be an idiot”):                                                    top to bottom:
(24) John considers *Fred *an idiot.                                 Type              Example
                                                                                    Proper noun         John
                                                                             Pronoun          He
  Note that the mention Fred, however, can be con-                   Deﬁnite NP            the man
nected to other mentions of Fred in the text.                               Indeﬁnite speciﬁc NP  a man I know                                                                                Non-speciﬁc NP     man
3.8  Temporal expressions
                                                    This leads to the following cases:  Temporal expressions such as the following are
linked:                                                     (28) Johnhead, a linguistattribute
(25) John spent three years in jail. In that time...     (29) A famous linguistattribute, hehead studied at ...

                                         7
<a id="page-8"></a>

### PDF 第 8 页

Type                Description
               Annotator Error    An annotator error. This is a catch-all category for cases of errors that do not ﬁt in the other
                                        categories.
              Genuine Ambiguity  This is just genuinely ambiguous. Often the case with pronouns that have no clear an-
                                      tecedent (especially this & that)
                Generics         One person thought this was a generic mention, and the other person didn’t
                Guidelines        The guidelines need to be clear about this example
                  Callisto Layout     Something to do with the usage/design of Callisto
                Referents          Each annotator thought this was referring to two completely different things
                Possessives       One person did not mark this possessive
               Verb            One person did not mark this verb
                Pre Modiﬁers      One person did not mark this Pre Modiﬁer
                Appositive        One person did not mark this appositive
              Copula             Disagreement arose because this mention is part of a copular structure
                                        a) Either each annotator marked a different half of the copula
                                      b) Or one annotator unnecessarily marked both

                              Figure 1: Description of various disagreement types.


                                                Annotator Error
                                        Genuine Ambiguity
                                                      Generics
                                                     Guidelines
                                                          Callisto Layout
                                                           Referents
                                                         Possessives
                                                         Verbs
                                                    Pre Modifiers
                                                       Appositives
                                                  Copulae

                                        0%     5%     10%     15%     20%     25%     30%


Figure 2: The distribution of disagreements across the various types in Table 1 for a sample of 15K disagreements in
the English portion of the corpus.


(30) a principal of the ﬁrmattribute, J. Smithhead     spans are linked.  In the example below, the en-
                                                                  tire span can be linked to later mentions to Richard
  In cases where the two members of the appositive   Godown.
are equivalent in speciﬁcity, the left-most member of     The sub-spans are not included separately in the
the appositive is marked as the head/referent. Deﬁ-   IDENT chain.
nite NPs include NPs with a deﬁnite marker (the) as
well as NPs with a possessive adjective (his). Thus   (35) Richard Godown, president of the Indus-
the ﬁrst element is the head in all of the following         trial Biotechnology Association
cases:
                                             Ages are tagged as attributes (as if they were el-
                                                          lipses of, for example, a 42-year-old):(31) The chairman, the man who never gives up
(32) The sheriff, his friend                           (36) Mr.Smithhead, 42attribute,
(33) His friend, the sheriff                                                       Similar rules apply for Chinese and Arabic. Un-
                                                           like English, where most appositives have a punctu-
  In the speciﬁcity scale, speciﬁc names of diseases                                                       ation marker, in Chinese that is not necessarily the
and technologies are classiﬁed as proper names,                                                     frequent case. In the following example we can see
whether they are capitalized or not.                                               an appositive construction without any punctuations
                                              between the head and the attribute.
(34) A dangerous bacteria, bacillium, is found
                                                     (37) 上图左起：(无锡市市长)X[attribute]
  When the entity to which an appositive refers is
also mentioned elsewhere, only the single span con-     (王宏民)X[head]，(副市长)Y[attribute]
taining the entire appositive construction is included     (洪锦、张怀西)Y[head]，...
in the larger IDENT chain. None of the nested NP

                                         8
<a id="page-9"></a>

### PDF 第 9 页

( H* ) hQå Language  Genre                     A1-A2  A1-ADJ  A2-ADJ      (40)  éJ k. PAmÌ'@  èP@Pð  Õæ @ H.  é ®£AJË@
 English   Newswire [NW]                   80.9     85.2     88.3       I Ë ( Aë )  à@    @  : É¯ñJ CJ K@X  éK Qå ñË@
           Broadcast News [BN]              78.6     83.5     89.4
           Broadcast Conversation [BC]       86.7     91.6     93.7                          ­JJk. ú ¯ B ð àQK. ú ¯          Magazine [MZ]                    78.4     83.2     88.8
          Weblogs and Newsgroups [WB]     85.9     92.2     91.2
           Telephone Conversation [TC]       81.3     94.1     84.7         The Swiss foreign ministry’s            Pivot Text [PT] (New Testament)    89.4     96.0     92.0
                                                   spokeswoman announced the (she) is Chinese          Newswire                  [NW]                                              73.6                                                       84.8                                                                75.1
           Broadcast                 News                         [BN]                                              80.5                                                       86.4                                                                91.6            neither in Burne nor in Geneva Pronouns
           Broadcast Conversation [BC]       84.1     90.7     91.2            in quoted speech are also marked.
          Magazine [MZ]                    74.9     81.2     80.0
          Weblogs and Newsgroups [WB]     87.6     92.3     93.5
           Telephone Conversation [TC]       65.6     86.6     77.1 3.11  Annotator Agreement and Analysis
                                                     Table 1 shows the inter-annotator and annotator-
Table 1: Inter Annotator (A1 and A2) and Adjudicator   adjudicator agreement on all the genres and lan-
(ADJ) agreement for the Coreference Layer in OntoNotes                                                guages of OntoNotes. A 15K disagreements in var-measured in terms of the MUC score.                                                     ious parts of the English data was analyzed, and
                                               grouped into one of the categories shown in Figure
                                                           1.  Figure 2 shows the distribution of these differ-
     Figure above from left : Wuxi                    ent types that were found in that sample.  It can be
     MayorX[attribute] Wang HongminX[head],        seen that genuine ambiguity and annotator error are
    Deputy MayorsY[attribute] Hong Jin, Zhang     the biggest contributors — the latter of which is usu-
     HuaixiY[head], ...                                    ally captured during adjudication, thus showing the
                                                     increased agreement between the adjudicated ver-
                                                     sion and the individual annotator version. Interest-
3.10  Special Issues                                    ingly, this mirrors the annotator disagreement analy-
  In addition to the ones above, there are some spe-    sis on the MUC corpus provided by Hirschman et al.
cial cases such as:                                    (1998).
                                        4  CoNLL-2012 Coreference Task
  • No coreference is marked between an organi-     The CoNLL-2012 shared task was held across all
     zation and its members.                          three languages — English, Chinese and Arabic —
  • GPEs are linked to references to their govern-   of the OntoNotes v5.0 data. The task was to auto-
     ments, even when the references are nested   matically identify mentions of entities and events in
     NPs, or the modiﬁer and head of a single NP.      text and to link the coreferring mentions together to
                                             form entity/event chains. The coreference decisions  • In extremely rare cases, metonymic mentions                                              had to be made using automatically predicted infor-    can be co-referenced. This is done only when                                                mation on other structural and semantic layers in-     the two mentions clearly and without a doubt                                                    cluding the parses, semantic roles, word senses, and     refer to the same entity. For example:                                         named entities. Given various factors, such as the
                                                     lack of resources and state-of-the-art tools, and time     (38) In a statement released this afternoon, 10
                                                          constraints, we could not provide some layers of in-       Downing Street called the bombings in
                                                  formation for the Chinese and Arabic portion of the         Casablanca “a strike against  all peace-
                                                          data.          loving people.”
                                             The three languages are from quite different lan-
     (39) In  a  statement,  Britain  called  the   guage families. The morphology of these languages
         Casablanca bombings “a strike against all    is quite different. Arabic has a complex morphol-
          peace-loving people.”                      ogy, English has limited morphology, whereas Chi-
                                                  nese has very little morphology. English word seg-
     In  this case,  it  is obvious that “10 Down-                                                  mentation amounts to rule-based tokenization, and
     ing Street” and “Britain” are being used inter-                                                                  is close to perfect. In the case of Chinese and Ara-
    changeably in the text. Again, if there is any                                                            bic, although the tokenization/segmentation is not
     ambiguity, however, these terms are not coref-                                                     as good as English, the accuracies are in the high
     erenced with each other.                                                      90s. Syntactically, there are many dropped subjects
  • In Arabic, verbal inﬂections are not considered   and objects in Arabic and Chinese, whereas English
    pronominal and are not coreferenced. The por-    is not a pro-drop language.  Another difference is
     tion marked with an * in the example below is   the amount of resources available for each language.
    an inﬂection and not a pronoun, and so should   English has probably the most resources at its dis-
     not be marked.                                    posal, whereas Chinese and Arabic lack signiﬁcantly

                                         9
<a id="page-10"></a>

### PDF 第 10 页

— Arabic more so than Chinese. Given this fact,   4.1.1  Closed Track
plus the fact that the CoNLL format cannot handle      In the closed track, systems were limited to the
multiple segmentations, and that it would compli-   provided data.  For the training and test data, in
cate scoring since we are using exact token bound-   addition to the underlying text, predicted versions
aries (as discussed later in Section 4.5), we decided   of all the supplementary layers of annotation were
to allow the use of gold, treebank segmentation for   provided using off-the-shelf tools (parsers, semantic
all languages.  In the case of Chinese, the words   role labelers, named entity taggers, etc.) retrained on
themselves are lemmas, so no additional informa-   the training portion of the OntoNotes data — as de-
tion needs to be provided. For Arabic, by default   scribed in Section 4.4.2. For the training data, how-
written text is unvocalised, so we decided to also   ever, in addition to predicted values for the other lay-
provide correct, gold standard lemmas, along with    ers, we also provided manual, gold-standard anno-
the correct vocalized version of the tokens. Table 2   tations for all the layers. Participants were allowed
lists which layers were available and quality of the   to use either the gold-standard or predicted annota-
provided layers (when provided.)                       tion to train their systems. They were also free to
                                                  use the gold-standard data to train their own models
                                                         for the various layers of annotation, if they judged
          Layer          English  Chinese  Arabic             that those would either provide more accurate pre-
           Segmentation        Lemma      √•   —•      •               dictions or alternative predictions for use as multiple                                         •              views, or if they wished to use a lattice of predic-            Parse       √    √    √6
            Proposition    √    √                        tions.            Predicate Frame  √         ×         Word Sense    √    √×   √×             More so than previous CoNLL shared  tasks,
        Name Entities   √                           coreference predictions depend on world knowl-
           Speaker           ×   —×              edge, and many state-of-the-art systems use infor-                          •       •
                                                mation from external resources such as WordNet,
Table 2: Summary of predicted layers provided for each   which provides a layer of information that could
language. A “•” indicates gold annotation, a “√” indi-   help a system recognize semantic connections be-
cates predicted, a “×” indicates an absence of the pre-   tween the various lexicalized mentions in the text.
dicted layer, and a “—” indicates that the layer is not ap-                                                     Therefore, in the case of English, similar to the pre-plicable to the language.                                                   vious year’s task, we allowed the use of WordNet in
                                                      the closed track. Since word senses in OntoNotes
                                                      are predominantly7  coarse-grained groupings  of  As is customary for CoNLL tasks, there were two                                           WordNet senses, systems could also map from theprimary tracks — closed and open. For the closed                                                     predicted or gold-standard word senses to the setstrack, systems were limited to using the distributed                                                     of underlying WordNet senses. Another signiﬁcantresources, in order to allow a fair comparison of al-                                                    piece of knowledge that is particularly useful forgorithm performance, while the open track allowed                                                    coreference but that is not available in the layers offor almost unrestricted use of external resources in                                              OntoNotes is that of number and gender. There areaddition to the provided data. Within each closed                                        many different ways of predicting these values, withand open track, we had an optional supplementary                                                         differing accuracies, so in order to ensure that par-track which allowed us to run some ablation studies                                                           ticipants in the closed track were working from theover a few different input conditions. This allowed                                           same data, thus allowing clearer algorithmic com-us to evaluate the systems given:  i) Gold mention                                                        parisons, we speciﬁed a particular table of numberboundaries (GB), ii) Gold mentions (GM), and iii)                                              and gender predictions generated by Bergsma andGold parses (GS). We will refer to the main task –                                                Lin (2006), for use during both training and test-where no mention boundaries are provided – as NB.                                                          ing. Unfortunately neither Arabic, nor Chinese have
4.1  Primary Evaluation                        comparable resources available that we could allow
                                                         participants to use. Chinese, in particular, does not  The primary evaluation comprises the closed and
                                                have number or gender inﬂections for nouns, butopen tracks where predicted information is provided
                                                 (Baran and Xue, 2011) look at a way to infer suchon all layers of the test set other than coreference. As
                                                     information.mentioned earlier, we provide gold lemma and vo-
calization information for Arabic, and we use gold   4.1.2  Open Track
standard treebank segmentation for all three lan-      In addition to resources available in the closed
guages.                                                   track, in the open track, systems were allowed to use

                                                              7There are a few instances of novel senses introduced in
   6The predicted part of speech for Arabic are a mapped down   OntoNotes which were not present in WordNet, and so lack a
version of the richer gold version present in the treebank         mapping back to the WordNet senses

                                        10
<a id="page-11"></a>

### PDF 第 11 页

Algorithm 1 Procedure used to create OntoNotes training, development
       and test partitions.
        Procedure: Generate Partitions(OntoNotes) returns Train, Dev, Test

            1: Train ←∅
            2: Dev ←∅
            3: Test ←∅
            4: for all Source ∈OntoNotes do            5:     if Source = Wall Street Journal then
            6:      Train ←Train ∪Sections 02 – 21
            7:     Dev ←Dev ∪Sections 00, 01, 22, 24
            8:     Test ←Test ∪Section 23            9:    else
          10:         if Number of ﬁles in Source ≥10 then
          11:        Train ←Train ∪File IDs ending in 1 – 8
          12:       Dev ←Dev ∪File IDs ending in 0
          13:        Test ←Test ∪File IDs ending in 9          14:       else
          15:       Dev ←Dev ∪File IDs ending in 0
          16:        Test ←Test ∪File ID ending in the highest number
          17:        Train ←Train ∪Remaining File IDs for the Source          18:     end if
          19:   end if
          20: end for
          21: return Train, Dev, Test



external resources such as Wikipedia, gazetteers etc.   tection and anaphoricity determination8. These also
The purpose of this track is mainly to get an idea   include potential spans that do not align with any
of the performance ceiling on the task at the cost of   constituent in the predicted parse tree.
not being able to perform a fair comparison across                                          Gold Parses (GS)  In this case, for each language,all systems. Another advantage of the open track is                                      we replaced the predicted parses in the closed trackthat it might reduce the barriers to participation by                                                     data with manual, gold parses.allowing participants to ﬁeld existing research sys-
tems that already depend on external resources —   4.3  Train, Development and Test Splits
especially if there were hard dependencies on these     For various reasons, not  all the documents in
resources — so they can participate in the task with   OntoNotes have been annotated with all the different
minimal, or no modiﬁcation to their existing system.   layers of annotation, with full coverage.9 There is a
                                                   core portion, however, which is roughly 1.6M En-4.2  Supplementary Evaluation                                                         glish words, 950K Chinese words, and 300K Arabic  In addition to the option of selecting between the                                             words which has been annotated with all the layers.primary closed or the open tracks, the participants                                                  This is the portion that we used for the shared task.also had an option to run their systems in the follow-                                   We used the same algorithm as in CoNLL-2011 toing ablation settings.
Gold Mention Boundaries (GB)  In this case, we      8Mention detection interacts with anaphoricity determina-
provided all possible correct mention boundaries in    tion since the corpus does not contain any singleton mentions.                                                         9As mentioned earlier, large scale manual annotation of var-the test data.  This essentially entails all NPs, and                                                                 ious layers of syntax and semantics is an expensive endeavor.
PRPs in the data extracted from the gold parse trees,   Adding to this, the fact that word sense annotation is most ef-
as well as the mentions that do not align with any    ﬁciently done one lemma at a time, ideally all instances of the
parse constituent, for example, non-existent con-   same across the entire corpus, or as large a portion as possi-
                                                                         ble, full coverage across all lemma instances is hard to achievestituents in the predicted parse owing to errors, some                                                              given the long tail of low frequency lemmas with a Zipﬁan dis-
named entities, etc.                                                tribution.  Similar issue affects PropBank annotation, but fur-
                                                               thermore, currently it only covers mostly verb predicates, and aGold Mentions (GM)  In this dataset, we provided                                                       few eventive noun predicates.
only and all the correct mentions for the test sets,       10http://projects.ldc.upenn.edu/ace/data/
thereby reducing the task to one of pure mention      11These numbers are for the part of OntoNotes v5.0 that have
clustering, and eliminating the task of mention de-     all layers of annotation including coreferenced.

                                        11
<a id="page-12"></a>

### PDF 第 12 页

Corpora          Language          Words                   Documents
                                                          Total  Train  Dev   Test     Total    Train   Dev    Test
                   MUC-6             English    25K  12K    13K         60      30      30
                   MUC-7             English    40K  19K    21K         67      30      37
                      ACE10(2000-2004)  English   960K 745K    215K               -           -          -
                                        Chinese   615K 455K    150K               -           -          -
                                         Arabic    500K 350K    150K               -           -          -
                      OntoNotes11       English   1.6M  1.3M 160K 170K   (3493)2,384   (2,802)1,940   (343)222   (348)222
                                        Chinese   950K 750K 110K  90K  (2,280)1,729   (1,810)1,391   (252)172   (218)166
                                         Arabic    300K 240K  30K  30K    (447)447     (359)359     (44)44     (44)44

Table 3: Number of documents in the OntoNotes v5.0 data, and some comparison with the MUC and ACE data sets. The
numbers in parenthesis for the OntoNotes corpus indicate the total number of parts that correspond to the documents.
Each part was considered a separate document for evaluation purposes.


create the train/development/test partitions for En-   4.4.1  Manual Annotation Gold Layers
glish, Chinese and Arabic. We tried to reuse pre-     Let us take a look at the manually annotated, or
viously established partitions for Chinese and Ara-   gold layers of information that were made available
bic, but either they were not in the selection used for   for the training data.
OntoNotes, or were partially overlapping, or had a
very small portion of OntoNotes covered in the test   Coreference  The manual coreference annotation
set.  Unfortunately, unlike English WSJ partitions,    is stored as chains of linked mentions connecting
there was no clean way of reusing those partitions.   multiple mentions of the same entity. Coreference is
Algorithm 1 details this procedure. The list of train-   the only document-level phenomenon in OntoNotes,
ing/development/test document IDs can be found on   and the complexity of annotation increases non-
the task webpage12. Following the recent CoNLL   linearly with the length of a document.  Unfortu-
tradition, participants were allowed to use both the    nately, some of the documents — especially the
training and the development data to train their ﬁnal   ones in the broadcast conversation, weblogs, and
model(s).                                          telephone conversation genre — are very long and
                                                           that prohibited efﬁcient annotation in their entirety.  The number of documents in the corpus for this
                                              These had to be split into smaller parts. A few passestask, for each of the different languages, and for
                                                         to join some adjacent parts were conducted, buteach of the training/development/test portions, are
                                                      since some documents had as many as 17 parts, thereshown in Table 3. For comparison purposes, it also
                                                      are still multi-part documents in the corpus. Sincelists the number of documents in the MUC-6, MUC-
                                                      the coreference chains are coherent only within each7, and ACE (2000-2004) corpora. The MUC-6 data
                                                     of these document parts, for the purpose of this task,was taken from the Wall Street Journal, whereas the
                                                each such part is treated as a separate document. An-MUC-7 data was from the New York Times. The ACE
                                                      other thing to note is that there were some casesdata spanned many different languages and genres
                                                     of sub-token annotation in the corpus owing to thesimilar to the ones in OntoNotes.  In fact, there is
                                                           fact that tokens were not split at hyphens.  Casessome overlap between ACE and OntoNotes source
                                                such as pro-WalMart had the sub-span WalMart linkeddocuments.
                                                  with another instance of WalMart. The recent Tree-
4.4  Data Preparation                          bank revision split tokens at most hyphens and made
  This section gives details of the different annota-   a majority of these sub-token annotations go away.
tion layers including the automatic models that were   There were  still some residual sub-token annota-
used to predict them, and describes the formats in    tions. Since subtoken annotations cannot be repre-
which the data was provided to the participants.       sented in the CoNLL format, and they were a very
                                                   small quantity — much less than even half a per-
   12http://conll.cemantix.org/2012/download/ids/               cent — we decided to ignore them. Unlike English,
                                                Chinese and Arabic have coreference annotation on
    For  each  language  there  are  two  sub-directories      elided subjects/objects. Recovering these entities in
  — “all” contains more general  lists which include       text is a hard problem, and the most recently re-    documents that had at least one of the layers of an-
     notation, and “coref” contains the  lists that include      ported numbers in literature for Chinese are around
    document that have coreference annotation. The former     a F-score of 50 (Yang and Xue, 2010; Cai et al.,
    were used to generate training/development/test sets      2011b). For Arabic there have not been much stud-
     for layers other than coreference, and the latter was       ies on recovering these. A study by Gabbard (2010)    used to generate training/development/test sets for the
    coreference layer used in this shared task.              shows that these can be recovered with an F-score

                                        12
<a id="page-13"></a>

### PDF 第 13 页

of 55 with automatic parses and roughly 65 using    Language  Type             Train   Development   Test     All
gold parses13.  Considering the level of prediction     English    Entities/Chains   35,143         4,546   4,532   44,221
accuracy of these tokens, and the relative frequency               Links          120,417       14,610  15,232  150,259                                                                           Mentions       155,560       19,156  19,764  194,480
of the same, plus the fact that the CoNLL tabular
format is not amenable to a variable number of to-     Chinese    Entities/Chains   28,257         3,875   3,559   35,691                                                                              Links           74,597       10,308   9,242   94,147
kens, we decided not to consider them as part of               Mentions       102,854       14,183  12,801  129,838
the task. In other words, we removed the manually     Arabic     Entities/Chains    8,330         936    980   10,246
identiﬁed traces (*pro* and *) respectively in Chi-               Links           19,260         2,381   2,255   23,896
nese and Arabic Treebanks. We also do not consider               Mentions        27,590         3,313   3,235   34,138
the links that are formed by these tokens in the gold
evaluation key.                                        Table 5: Number of entities, links and mentions in the
                                                  OntoNotes v5.0 data.  Tables 4 and 5 shows the distribution of mentions
by the syntactic categories, and the counts of enti-
ties, links and mentions in the corpus respectively.
Interestingly the mentions formed by these dropped    that are legitimate targets for coreference annota-
pronouns total roughly about 11% for both Chinese    tion.  Function tags were also removed, since the
and Arabic.  All of this data has been Treebanked   parsers that we used for the predicted syntax layer
and PropBanked either as part of the OntoNotes ef-   did not provide them. One thing that needs to be
fort, or some previous effort.                           dealt with in conversational data is the presence of
                                                    disﬂuencies (restarts, etc.).  In the English parses
                                                     of the OntoNotes, the disﬂuencies are marked using
 Language            Syntactic                              Train                                    Development                                                              Test
            category                       Count                  %   Count                          %   Count                                 %   a special EDITED14 phrase tag — as was the case
 English   Noun Phrase  61.8K  39.46  9.7K  45.57  9.2K  42.97  for the Switchboard Treebank. Given the frequency
          Pronoun      66.7K  42.61  7.8K  36.66  8.2K  38.69  of disﬂuencies and the performance with which one
           Proper              Noun                       18.1K                                11.60                                    2.2K                                              10.66                                                 2.3K                                                            10.96
          Dropped                      Pro.                                        -                                                 -                                                           -                                                                    -                                                                             -                                                                                      -  can identify them automatically,15 a probable pro-
           Other Noun    2.636   1.68   546   2.55   500   2.33  cessing pipeline would ﬁlter them out before pars-
           Verb          2.522   1.61   299   1.40   342   1.60  ing. Since we did not have a readily available tag-           Other         4.761   3.04   676   3.16   738   3.45
                                                    ger for tagging disﬂuencies, we decided to remove
 Chinese         Noun Phrase                       40.7K                                34.23                                    5.4K                                              32.53                                                 5.1K                                                            35.31  them using oracle information available in the En-          Pronoun                       20.8K                                17.50                                    3.3K                                              19.88                                                 2.5K                                                            17.65
          Dropped Pro.  13.5K  11.39  1.9K  12.04  1.5K  10.71  glish Treebank, and the coreference chains were
           Proper Noun  19.0K  15.96  2.8K  17.24  2.2K  15.54  remapped to trees without disﬂuencies. Owing to           Other Noun   23.6K  19.88  2.8K  17.08  2.8K  19.71
           Verb          244   0.20    51   0.31    20   0.14  various constraints, we decided to retain the disﬂu-
           Other          994   0.83   153   0.92   139   0.95  encies in the Chinese data. Since Arabic portion of
 Arabic    Noun Phrase  10.8K  34.93  1.3K  35.02  1.3K  36.51  the corpus is all newswire, this had no impact on
          Pronoun       8.9K  28.77  1.0K  28.33  1.1K  30.58   it. However, for both Chinese and Arabic, since we          Dropped Pro.   3.5K  11.52   477  12.57   429  11.78
           Proper Noun   4.0K  13.01   450  11.86   390  10.71  remove trace tokens corresponding to dropped pro-
           Other Noun    3.3K  10.90   439  11.57   345   9.47  nouns, all the other layers of annotation had to be           Verb           25   0.08     4   0.11     0   0.00
           Other          247   0.79    21   0.55    35   0.96  remapped to the remaining sequence of tree tokens.
                                                  Propositions  The propositions in OntoNotes are
Table 4: Distribution of mentions in the data by their syn-   PropBank-style semantic roles for English, Chinese
tactic category.                                              and Arabic. Most of the verb predicates in the cor-
                                                pus have been annotated with their arguments. As
                                                         part of the OntoNotes effort, some enhancements
Parse Trees  These represent the syntactic layer                                              were made to the English PropBank and Treebank
that is a revised version of the treebanks in English,                                                         to make them synchronize better with each other
Chinese and Arabic. Arabic treebank has probably                                              (Babko-Malaya et al., 2006). One of the outcomes
seen the most revision over the past few years, in an                                                     of this effort was that two types of LINKs that rep-
effort to increase consistency. For purposes of this                                                        resent pragmatic coreference (LINK-PCR) and selec-
task, traces were removed from the syntactic trees,
since the CoNLL-style data format, being indexed      14There is another phrase type — EMBED in the telephone
by tokens, does not provide any good means of con-    conversation genre which is similar to the EDITED phrase type,
veying that information. As mentioned in the previ-   and sometimes identiﬁes insertions, but sometimes contains                                                                     logical continuation of phrases by different speakers, so we de-ous section, these include the cases of traces in Chi-                                                              cided not to remove that from the data.
nese and Arabic which are dropped subjects/objects      15A study by Charniak and Johnson (2001) shows that one
                                                          can identify and remove edits from transcribed conversational
  13These numbers are not in the thesis, but we received them    speech with an F-score of about 78, with roughly 95 Precision
in an email communication with the Ryan Gabbard.             and 67 recall.

                                        13
<a id="page-14"></a>

### PDF 第 14 页

tional preferences (LINK-SLC) were added to Prop-         Layer             English    Chinese    Arabic
Bank. More details can be found in the addendum to                        Verb Noun    All   Verb Noun
the PropBank guidelines16 in the OntoNotes v5.0 re-          Sense Inventories  2702  2194     763   150   111
lease. Since the community is not used to this repre-         Frames          5672  1335   20134  2743   532
sentation which relies heavily on the trace structure                                                       Table 7: Number of senses deﬁned for English, Chinesein the Treebank which we are excluding, we decided                                                  and Arabic in the OntoNotes v5.0 corpus.
to unfold the LINKs back to their original represen-
tation as in the PropBank 1.0 release. This function-
ality is part of the OntoNotes DB Tool.17
Word Sense  Gold standard word sense annotation   for that layer from the training portion of the entire
was supplied using sense numbers (along with the   OntoNotes v5.0 release.
sense inventories) as speciﬁed in the OntoNotes list
of senses for each lemma. The coverage of the word   Parse Trees  Predicted parse trees for English were
sense annotation varies among the languages. En-   produced using the Charniak parser18 (Charniak and
glish has the most coverage, while coverage for Chi-   Johnson, 2005).  Some additional tag types used
nese and Arabic is more sporadic. Even for English,   in the OntoNotes trees were added to the parser’s
the coverage for word sense annotation is not com-    tagset, including the NML tag that has recently been
plete. Only some of the verbs and nouns are anno-   added to capture internal NP structure, and the rules
tated with word sense information.                  used to determine head words were extended corre-
                                                     spondingly. Chinese and Arabic parses were gen-Named Entities  Named  Entities  in OntoNotes
                                                      erated using the Berkeley parser (Petrov and Klein,data are speciﬁed using a catalog of 18 Name types.
                                                   2007). In the case of Arabic, the parsing commu-
Other Layers  Discourse plays a  vital  role  in   nity uses a mapping from rich Arabic part of speech
coreference resolution. In the case of broadcast con-    tags, to Penn-style part of speech tags. We used the
versation, or telephone conversation data, it partially   mapping that is included with the Arabic treebank.
manifests itself in the form of speakers of a given ut-                                             The predicted parses for the training portion of
terance, whereas in weblogs or newsgroups it does                                                      the data were generated using 10-fold (5-fold for
so as the writer, or commenter of a particular article                                                   Arabic) cross-validation. The development and test
or thread. This information provides an important                                                     parses were generated using a model trained on the
clue for correctly linking anaphoric pronouns with                                                          entire training portion. We used OntoNotes v5.0
the right antecedents. This information could be au-                                                         training data for training the Chinese and Arabic
tomatically deduced, but since it would add addi-                                                      parser models, but the OntoNotes v4.0 subset of
tional complexity to the already complex task, we                                              OntoNotes v5.0 data was used for training the En-
decided to provide oracle information of this meta-                                                         glish model. We decided to do the latter to be able to
data both during training and testing. In other words,                                                          better compare the scores to the CoNLL-2011 eval-
speaker and author identiﬁcation was not treated as                                                     uation given that parser is a central component to a
an annotation layer that needed to be predicted. This                                                    coreference system, and the fact that OntoNotes v5.0
information was provided in the form of another col-                                                adds a small fraction of gold parses on top of those
umn in the .conll ﬁle. There were some cases of                                                  provided by OntoNotes v4.0. Table 6 shows the per-
interruptions and interjections that led to a sentence                                               formance of the re-trained parsers on the CoNLL-
associated with two different speakers, but since the                                            2012 test set. We did not get a chance to re-train the
frequency of this was quite small, we decided to                                                      re-ranker available for English, and since the stock
make an assumption of one speaker/writer per sen-                                                      re-ranker crashes when run on n-best parses contain-
tence.                                                    ing NMLs, because it has not seen that tag in train-
4.4.2  Predicted Annotation Layers                  ing, we could not make use of it. In addition to the
  The predicted annotation layers were derived us-   parser scores and part of speech accuracy, we have
ing automatic models trained using cross-validation   also added a column for the accuracy for the NPs
on other portions of OntoNotes v5.0 data.  As   because they are particularly relevant to the corefer-
mentioned earlier, there are some portions of the   ence task.
OntoNotes corpus that have not been annotated for
coreference but that have been annotated for other
layers. For training models for each of the layers,       18http://bllip.cs.brown.edu/download/reranking-
where feasible, we used all the data that we could    parserAug06.tar.gz                                                                19There was an error in processing the test set, therefore the
                                                           performance on the test set was slightly lower than the correct
   16doc/propbank/english-propbank.pdf                       one reported in the table. The performance of the sense tagging
   17http://cemantix.org/ontonotes.html                             the ofﬁcal test set is 77.6 (R), 71.5 (P) and 74.4 (F).

                                        14
<a id="page-15"></a>

### PDF 第 15 页

All Sentences                 Sentence length < 40
                                  N   POS  NP   R    P    F   N   R    P    F
             English  Broadcast Conversation [BC]     2,194  95.93  90.05  84.30  84.46  84.38  2,124  85.83  85.97  85.90
                      Broadcast News [BN]            1,344  96.50  91.11  84.19  84.28  84.24  1,278  85.93  86.04  85.98
                    Magazine [MZ]                 780  95.14  91.63  87.11  87.46  87.28   736  87.71  88.04  87.87
                   Newswire [NW]                 2,273  96.95  90.14  87.05  87.45  87.25  2,082  88.95  89.27  89.11
                     Telephone Conversation [TC      1,366  93.52  88.96  79.73  80.83  80.28  1,359  79.88  80.98  80.43
                    Weblogs and Newsgroups [WB]   1,658  94.67  89.16  83.32  83.20  83.26  1,566  85.14  85.07  85.11
                        Pivot Text [PT] (New Testament)  1,217  96.87  95.39  92.48  93.66  93.07  1,217  92.48  93.66  93.07
                       Overall                        9,615  96.03  90.78  85.25  85.43  85.34  9145  86.86  87.02  86.94
             Chinese  Broadcast Conversation [BC]      885  94.79  86.32  79.35  80.17  79.76   824  80.92  81.86  81.38
                      Broadcast News [BN]            929  93.85  86.00  80.13  83.49  81.78   756  81.82  84.65  83.21
                    Magazine [MZ]                 451  97.06  92.40  83.85  88.48  86.10   326  85.64  89.80  87.67
                   Newswire [NW]                 481  94.07  79.70  77.28  82.26  79.69   406  79.06  83.84  81.38
                     Telephone Conversation [TC]      968  92.22  80.15  69.19  71.90  70.52   942  69.59  72.24  70.89
                    Weblogs and Newsgroups [WB]    758  92.37  85.60  78.92  82.57  80.70   725  79.30  83.10  81.16
                       Overall                        4,472  94.12  85.74  78.93  82.23  80.55  3,979  79.80  82.79  81.27
             Arabic  Newswire [NW]                 1,003  94.12  80.70  75.67  74.71  75.19   766  77.44  74.99  76.19

                           Table 6: Parser performance on the CoNLL-2012 test set.


                                                                            Accuracy
                                                     R   P   F
                                        English  Broadcast Conversation [BC]    81.3  81.2  81.2
                                                Broadcast News [BN]           81.5  82.0  81.7
                                           Magazine [MZ]                 78.8  79.1  79.0
                                          Newswire [NW]                85.7  85.7  85.7
                                           Weblogs and Newsgroups [WB]  77.6  77.5  77.5
                                                  Overall                        82.5  82.5  82.5
                                      Chinese  Broadcast Conversation [BC]          -       -  80.5
                                                Broadcast News [BN]                   -       -  85.4
                                           Magazine [MZ]                          -       -  82.4
                                          Newswire [NW]                         -       -  89.1
                                                  Overall                                    -       -  84.3
                                      Arabic  Newswire [NW]19              75.2  75.9  75.6

             Table 8: Word sense performance over both verbs and nouns in the CoNLL-2012 test set.


Word Sense  This year we used the IMS (It Makes   identiﬁed nouns and verbs. In case of Arabic, IMS
Sense) (Zhong and Ng, 2010) word sense tagger.20   uses gold lemmas. Since automatic POS tagging is
Word sense information, unlike syntactic parse in-   not perfect, IMS does not always output a sense to
formation is not central to approaches taken by cur-    all word tokens that need to be sense tagged due to
rent coreference systems and so we decided to use   wrongly predicted POS tags. As such, recall is not
a better word sense tagger to get a good state of the   the same as precision on the English and Arabic test
art accuracy estimate, at the cost of a completely fair   data.  Recall that in Chinese, the word senses are
(but,  still close enough) comparison with English   deﬁned against lemmas and are independent of the
CoNLL-2011 results. This will also allow potential   part of speech.  Since we provide gold word seg-
future uses to beneﬁt from it. IMS was trained on   mentation, IMS attempts to sense tag all correctly
all the word sense data that is present in the train-   segmented Chinese words, so recall and precision
ing portion of the OntoNotes corpus using cross-   are same and so is F1. Table 7 gives the number of
validated predictions on the input layers similar to   lemmas covered by the word sense inventory in the
the proposition tagger. During testing, for English   English, Chinese and Arabic portion of OntoNotes.
and Arabic, IMS must ﬁrst uses the automatic POS
information to identify the nouns and verbs in the     Table 8 shows the performance of this classiﬁer
test data, and then assign senses to the automatically   aggregated over both the verbs and nouns in the
                                         CoNLL-2012 test set. For English, genres PT and
                                                 TC, and for Chinese genres TC and WB, no gold stan-  20We offer special thanks to Hwee Tou Ng and his student
Zhi Zhong for training IMS models and providing output for   dard senses were available, and so their accuracies
the development and test sets.                            could not be computed.

                                        15
<a id="page-16"></a>

### PDF 第 16 页

Propositions  We used ASSERT21 (Pradhan et al.,                          Framesets Lemmas
2005) to predict the propositional structure for En-                               1    2,722
glish.   Similar to the parser model for English,                               2     321
the same proposition model that was used in the                 > 2     181
CoNLL-2011 shared task — trained on all the train-                                                            Table 10: Frameset polysemy across lemmas.ing portion of the OntoNotes v4.0 data using cross-
validated predicted parses — was used to generate
the propositions for the development and test sets
for this evaluation. We took a two stage approach   chronize well with the Treebank, and ﬁnally v) it in-
to tagging where The NULL arguments are ﬁrst ﬁl-   cludes propositions for be verbs missing from the
tered out, and the remaining NON-NULL arguments   original PropBank.  It looks like the newly added
are classiﬁed into one of the argument types. The   Pivot Text data (comprising of the New Testament)
argument identiﬁcation module used an ensemble   shows very good performance. This is not surprising
of ten classiﬁers — each trained on a tenth of the   given a similar trend in it parsing performance.
training data and combined using unweighted vot-      In addition to automatically predicting the argu-
ing.  This should still give a close to state-of-the-   ments, we also trained a classiﬁer to tag PropBank
art performance given that the argument identiﬁ-   frameset IDs for the English data.  Table 7 lists
cation performance tends to start to be asymptotic   the number of framesets available across the three
around 10K training instances (Pradhan et al., 2005).   languages22 An overwhelming number of them are
The Chinese propositional structure was predicted   monosemous, but the more frequent verbs tend to be
with the Chinese semantic role labeler described in   polysemous. Table 10 gives the distribution of num-
(Xue, 2008), retrained on all the training portion of   ber of framesets per lemma in the PropBank layer of
the OntoNotes v5.0 data. No propositional struc-   the English OntoNotes v5.0 data.
tures were provided for Arabic due to resource con-     During automatic processing of the  data, we
straints.  Table 9 shows the detailed performance   tagged all the tokens that were tagged with a part
numbers. The CoNLL-2005 scorer was used to com-   of speech VBx. This means that there would be cases
pute the scores. At ﬁrst glance, the performance on   where the wrong token would be tagged with propo-
the English newswire genre is much lower than what    sitions.
has been reported for WSJ Section 23. This could                                    Named  Entities  BBN’s  IdentiFinderTMsystem
be attributed to several factors:  i) the fact that we                                           was used to predict the named entities.  For the
had to compromise on the training method, ii) the                                         CoNLL-2011 shared task we did not get a chance
newswire in OntoNotes not only contains WSJ data,                                                         to re-train Identiﬁnder, and used the stock model
but also Xinhua news, iii) The WSJ training and test                                             which did not have the same set of named entities
portions in OntoNotes are a subset of the standard                                                     as  in the OntoNotes corpus,  so we decided  to
ones that have been used to report performance ear-
lier; iv) the PropBank guidelines were signiﬁcantly      22The number of lemmas for English in Table 10 do not add
revised during the OntoNotes project in order to syn-   up to this number because not all of them have examples in
                                                                  the training data, where the total number of instantiated senses
   21http://cemantix.org/assert.html                           amounts to 4229.


                                                   Frameset    Total        Total   % Perfect   Argument ID + Class
                                                 Accuracy  Sentences  Propositions  Propositions   P    R     F
              English  Broadcast Conversation [BC]         92      2,037        5,021        52.18  82.55  64.84  72.63
                      Broadcast News [BN]               91      1,252        3,310        53.66  81.64  64.46  72.04
                    Magazine [MZ]                    89      780        2,373        47.16  79.98  61.66  69.64
                    Newswire [NW]                    93      1,898        4,758        39.72  80.53  62.68  70.49
                      Telephone Conversation [TC]         90      1,366        1,725        45.28  79.60  63.41  70.59
                    Weblogs and Newsgroups [WB]       92      929        2,174        39.19  81.01  60.65  69.37
                        Pivot Corpus [PT]                  92      1,217        2,853        50.54  86.40  72.61  78.91
                       Overall                           91      9,479       24,668        44.69  81.47  61.56  70.13
             Chinese  Broadcast Conversation [BC]                -      885        2,323        31.34  53.92  68.60  60.38
                      Broadcast News [BN]                         -      929        4,419        35.44  64.34  66.05  65.18
                    Magazine [MZ]                                -      451        2,620        31.68  65.04  65.40  65.22
                    Newswire [NW]                                -      481        2,210        27.33  69.28  55.74  61.78
                      Telephone Conversation [TC]                -      968        1,622        32.74  48.70  59.12  53.41
                    Weblogs and Newsgroups [WB]             -      758        1,761        35.21  62.35  68.87  65.45
                       Overall                                          -      4,472       14,955        32.62  61.26  64.48  62.83

               Table 9: Performance on the propositions and framesets in the CoNLL-2012 test set.


                                        16
<a id="page-17"></a>

### PDF 第 17 页

All Genre   BC   BN  MZ  NW   TC  WB
                                            F      F     F     F    F    F     F
                             English  Cardinal         68.76   58.52   75.34  72.57  83.62  32.26   57.14
                                   Date             78.60   73.46   80.61  71.60  84.12  63.89   65.48
                                    Event            44.63   30.77   50.00  36.36  50.00   0.00   66.67
                                            Facility          47.29   64.20   43.14  40.00  54.17   0.00   28.57
                               GPE             89.77   89.40   93.83  92.87  92.56  81.19   91.36
                                  Language        47.06          -   75.00  50.00  33.33  22.22   66.67
                            Law             48.00    0.00  100.00   0.00  50.98   0.00  100.00
                                     Location         59.00   54.55   61.36  54.84  67.10        -   44.44
                             Money           75.45   33.33   63.64  77.78  79.12  92.31   58.18
                             NORP            88.58   94.55   93.92  94.87  90.70  78.05   85.15
                                      Ordinal          71.39   74.16   80.49  79.07  74.34  84.21   55.17
                                      Organization     76.00   60.90   78.57  69.97  84.76  48.98   51.08
                                       Percent          89.11  100.00   83.33  75.00  91.41  83.33   72.73
                                    Person           78.75   93.35   94.36  87.47  85.80  73.39   76.49
                                     Product          52.76    0.00   77.65   0.00  42.55   0.00    0.00
                                      Quantity         50.00   17.14   66.67  62.86  81.82   0.00   30.77
                                Time            60.65   66.13   67.33  66.67  64.29  27.03   55.56
                              Work of Art      34.03   42.42   35.62  28.57  54.24   0.00    8.70
                                       Overall          77.95   77.02   84.95  80.33  84.73  62.17   69.47

             Table 11: Named Entity performance on the English subset of the CoNLL-2012 test set.


update the model for this round by retraining  it    sites  that  already had  tools conﬁgured  to  deal
on  the English  portion  of  the OntoNotes v5.0   with that format.  Therefore, in order to distribute
corpus. Given the various constraints, we could not   the data so that one could make the best of both
re-train  it on the Chinese and Arabic data, Table   worlds, we created a new ﬁle extension — .conll
11 shows the overall performance of the tagger   which logically served as another layer in addition
on the CoNLL-2012 English test set, as well as   to the .parse,  .prop,  .sense,  .name and .coref
the performance broken down by individual name   layers which  house  the  respective  annotations.
types.                                        Each .conll ﬁle contained a merged representation
                                                     of  all the OntoNotes layers in the CoNLL-style
Other Layers  As noted earlier, systems were al-   tabular format with one line per token, and with
lowed to make use of gender and number predic-   multiple columns for each token specifying the
tions for NPs using the table from Bergsma and Lin   input annotation layers relevant to that token, with
(Bergsma and Lin, 2006), and the speaker meta-   the ﬁnal column specifying the target coreference
data for broadcast conversations, telephone conver-    layer. Because we are not authorized to distribute
sations and author or poster metadata for weblogs   the underlying text, and many of the layers contain
and newsgroups.                                         inline annotation, we had to provide a skeletal form
                                                      (.skel) of the .conll ﬁle which is essentially the4.4.3  Data Format                                                  .conll ﬁle, but with the column that contains the
  In order to organize the multiple, rich layers of                                                 words, anonymized.  We provided an assembly
annotation,  the OntoNotes project has created a                                                            script that participants could use to create a .conll
database representation for the raw annotation layers                                                    ﬁle taking as input the .skel ﬁle and the top-level
along with a Python API to manipulate them (Prad-                                                       directory of the OntoNotes distribution that they
han et al., 2007a). In the OntoNotes distribution the                                              had separately downloaded from the LDC23. Once
data is organized as one ﬁle per layer, per document.                                                      the .conll ﬁle is created,  it can be used to create
The API requires a certain hierarchical structure                                                      the individual layers such as .parse, .name, and
with various annotation layers represented by ﬁle                                                  .coref that have inline annotation, with the provided
extensions for the documents at the leaves, and                                                              scripts. We provide the layers that have standoff
language, genre, source and section within a partic-                                                    annotation (mostly with respect to the tokens in the
ular source forming the intermediate directories —                                                     treebank) like the .prop and .sense along with the
data/<language>/annotations/<genre>/<source>/                                                  .skel ﬁle.
<section>/<document>.<layer>.      It comes  with                                                        In the CoNLL-2011 task, there were a few issues,various ways of querying and manipulating the data                                             where some teams used the test data accidentallyand allows convenient access to the information                                                   during training. To prevent it from happening againinside the sense inventory and PropBank frame
ﬁles instead of having to interpret the raw .xml.                                                             23OntoNotes is deeply grateful to the Linguistic Data Con-
However,  maintaining  format  consistency  with    sortium for making the source data freely available to the task
earlier CoNLL tasks was deemed convenient for    participants.

                                        17
<a id="page-18"></a>

### PDF 第 18 页

Column                Type  Description
         1             Document ID  This is a variation on the document ﬁlename
         2                   Part number Some ﬁles are divided into multiple parts numbered as 000, 001, 002, ... etc.
         3             Word number  This is the word index in the sentence
         4                   Word The word itself
         5                 Part of Speech  Part of Speech of the word
         6                     Parse bit  This is the bracketed structure broken before the ﬁrst open parenthesis in the parse, and the
                                           word/part-of-speech leaf replaced with a *. The full parse can be created by substituting
                                            the asterisk with the ([pos] [word]) string (or leaf) and concatenating the items in
                                            the rows of that column.
         7               Lemma The predicate/sense lemma is mentioned for the rows for which we have semantic role or
                                   word sense information. All other rows are marked with a -
         8        Predicate Frameset ID  This is the PropBank frameset ID of the predicate in Column 7.
         9              Word sense  This is the word sense of the word in Column 4.
         10            Speaker/Author  This is the speaker or author name where available. Mostly in Broadcast Conversation and
                                   Weblog data.
         11          Named Entities  These columns identiﬁes the spans representing various named entities.
         12:N      Predicate Arguments  There is one column each of predicate argument structure information for the predicate
                                       mentioned in Column 7.
      N                Coreference  Coreference chain information encoded in a parenthesis structure.

                           Table 12: Format of the .conll ﬁle used in the shared task.





#begin document (nw/wsj/07/wsj_0771); part 000
...
...
nw/wsj/07/wsj_0771 0   0         ‘‘   ‘‘  (TOP(S(S*          -    -  -   -         *       *     (ARG1*          *         *      -
nw/wsj/07/wsj_0771 0   1 Vandenberg  NNP       (NP*          -    -  -   -  (PERSON)  (ARG1*          *          *         * (8|(0)
nw/wsj/07/wsj_0771 0   2        and   CC          *          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0   3    Rayburn  NNP         *)          -    -  -   -  (PERSON)      *)          *          *         *(23)|8)
nw/wsj/07/wsj_0771 0   4        are  VBP       (VP*         be   01  1   -         *    (V*)          *          *         *      -
nw/wsj/07/wsj_0771 0   5     heroes  NNS   (NP(NP*)          -    -  -   -         *  (ARG2*          *          *         *      -
nw/wsj/07/wsj_0771 0   6         of   IN       (PP*          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0   7       mine   NN   (NP*))))          -    -  5   -         *      *)          *          *         *   (15)
nw/wsj/07/wsj_0771 0   8          ,    ,          *          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0   9         ’’   ’’         *)          -    -  -   -         *       *         *)          *         *      -
nw/wsj/07/wsj_0771 0  10        Mr.  NNP       (NP*          -    -  -   -         *       *     (ARG0*     (ARG0*         *    (15
nw/wsj/07/wsj_0771 0  11      Boren  NNP         *)          -    -  -   -  (PERSON)       *         *)         *)         *    15)
nw/wsj/07/wsj_0771 0  12       says  VBZ       (VP*        say   01  1   -         *       *       (V*)          *         *      -
nw/wsj/07/wsj_0771 0  13          ,    ,          *          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0  14  referring  VBG     (S(VP*      refer   01  2   -         *       * (ARGM-ADV*       (V*)         *      -
nw/wsj/07/wsj_0771 0  15         as   RB     (ADVP*          -    -  -   -         *       *          * (ARGM-DIS*         *      -
nw/wsj/07/wsj_0771 0  16       well   RB         *)          -    -  -   -         *       *          *         *)         *      -
nw/wsj/07/wsj_0771 0  17         to   IN       (PP*          -    -  -   -         *       *          *     (ARG1*         *      -
nw/wsj/07/wsj_0771 0  18        Sam  NNP    (NP(NP*          -    -  -   -  (PERSON*       *          *          *         *    (23
nw/wsj/07/wsj_0771 0  19    Rayburn  NNP         *)          -    -  -   -        *)       *          *          *         *      -
nw/wsj/07/wsj_0771 0  20          ,    ,          *          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0  21        the   DT    (NP(NP*          -    -  -   -         *       *          *          *    (ARG0*      -
nw/wsj/07/wsj_0771 0  22 Democratic   JJ          *          -    -  -   -    (NORP)       *          *          *         *      -
nw/wsj/07/wsj_0771 0  23      House  NNP          *          -    -  -   -     (ORG)       *          *          *         *      -
nw/wsj/07/wsj_0771 0  24    speaker   NN         *)          -    -  -   -         *       *          *          *        *)      -
nw/wsj/07/wsj_0771 0  25        who   WP (SBAR(WHNP*)        -    -  -   -         *       *          *          * (R-ARG0*)      -
nw/wsj/07/wsj_0771 0  26 cooperated  VBD     (S(VP*  cooperate   01  1   -         *       *          *          *      (V*)      -
nw/wsj/07/wsj_0771 0  27       with   IN       (PP*          -    -  -   -         *       *          *          *    (ARG1*      -
nw/wsj/07/wsj_0771 0  28  President  NNP       (NP*          -    -  -   -         *       *          *          *         *      -
nw/wsj/07/wsj_0771 0  29 Eisenhower  NNP *)))))))))))        -    -  -   -  (PERSON)       *         *)         *)        *)    23)
nw/wsj/07/wsj_0771 0  30          .    .        *))          -    -  -   -         *       *          *          *         *      -

nw/wsj/07/wsj_0771 0   0         ‘‘   ‘‘    (TOP(S*          -    -  -   -         *       *          *          -
nw/wsj/07/wsj_0771 0   1       They  PRP      (NP*)          -    -  -   -         * (ARG0*)          *        (8)
nw/wsj/07/wsj_0771 0   2    allowed  VBD       (VP*      allow   01  1   -         *    (V*)          *          -
nw/wsj/07/wsj_0771 0   3       this   DT     (S(NP*          -    -  -   -         *  (ARG1*     (ARG1*         (6
nw/wsj/07/wsj_0771 0   4    country   NN         *)          -    -  3   -         *       *         *)         6)
nw/wsj/07/wsj_0771 0   5         to   TO       (VP*          -    -  -   -         *       *          *          -
nw/wsj/07/wsj_0771 0   6         be   VB       (VP*         be   01  1   -         *       *       (V*)       (16)
nw/wsj/07/wsj_0771 0   7   credible   JJ (ADJP*)))))         -    -  -   -         *      *)    (ARG2*)          -
nw/wsj/07/wsj_0771 0   8          .    .        *))          -    -  -   -         *       *          *          -
#end document

                                  Figure 3: Sample portion of the .conll ﬁle.





                                        18
<a id="page-19"></a>

### PDF 第 19 页

this year, we were advised by the steering commit-   unrestricted set of entity types:  i) The link based
tee to distribute the data in two installments. One for  MUC metric (Vilain et al., 1995), ii) The mention
training and development and the other for testing.   based B-CUBED metric (Bagga and Baldwin, 1998)
The test data released from LDC did not contain the   and iii) The entity based CEAF (Constrained En-
coreference layer. Therefore, this year unlike previ-    tity Aligned F-measure) metric (Luo, 2005). Very
ous CoNLL tasks, the test data contained some truly   recently BLANC (BiLateral Assessment of Noun-
unseen documents. This made it easier to spot po-   Phrase Coreference) measure (Recasens and Hovy,
tential training errors such as ones that occurred in   2011) has been proposed as well. Each metric tries
the CoNLL-2011 task. Table 12 describes the data   to address the shortcomings or biases of the earlier
provided in each of the column of the .conll format.   metrics. Given a set of key entities K, and a set of
Figure 3 shows a sample from a .conll ﬁle.           response entities R, with each entity comprising one
                                                     or more mentions, each metric generates its variation4.5  Evaluation
                                                     of a precision and recall measure. The MUC measure  This section describes the evaluation criteria used                                                                  is the oldest and most widely used.  It focuses onfor the shared task. Unlike propositions, word sense
                                                      the links (or, pairs of mentions) in the data.24 Theand named entities, where it is simply a matter of
                                           number of common links between entities in K andcounting the correct answers, or for parsing, where
                          R divided by the number of links in K representsthere is an established metric, evaluating the accu-
                                                      the recall, whereas, precision is the number of com-racy of coreference continues to be contentious. Var-
                                     mon links between entities in K and R divided byious alternative metrics have been proposed, as men-
                                                      the number of links in R. This metric prefers sys-tioned below, which weight different features of a
                                               tems that have more mentions per entity — a sys-proposed coreference pattern differently. The choice
                                            tem that creates a single entity of all the mentionsis not clear in part because the value of a particular
                                                         will get a 100% recall without signiﬁcant degrada-set of coreference predictions is integrally tied to the
                                                         tion in its precision. And, it ignores recall for single-consuming application. A further issue in deﬁning a
                                                    ton entities, or entities with only one mention. Thecoreference metric concerns the granularity of the
                                        B-CUBED metric tries to addresses MUC’s shortcom-mentions, and how closely the predicted mentions
                                                          ings, by focusing on the mentions and computes re-are required to match those in the gold standard for
                                                             call and precision scores for each mention.  If K isa coreference prediction to be counted as correct.
                                                      the key entity containing mention M, and R is the re-Our evaluation criterion was in part driven by the
                                                 sponse entity containing mention M, then recall forOntoNotes data structures. OntoNotes coreference
makes the distinction between identity coreference   the mention M is computed as |K∩R| and precision
                                                                                                       |K|and appositive coreference, treating the latter sepa-                                                         for the same is is computed as |K∩R| . Overall recall
rately. Thus we evaluated systems only on the iden-                                           |R|
tity coreference task, which links all categories of   and precision are the average of the individual men-
entities and events together into equivalent classes.   tion scores. CEAF aligns every response entity with
The situation with mentions for OntoNotes is also    at most one key entity by ﬁnding the best one-to-one
different than it was for MUC or ACE. OntoNotes   mapping between the entities using an entity simi-
data does not explicitly identify the minimum ex-    larity metric. This is a maximum bipartite matching
tents of an entity mention, but it does include hand-   problem and can be solved by the Kuhn-Munkres
tagged syntactic parses. Thus for the ofﬁcial evalua-   algorithm. This is thus a entity based measure. De-
tion, we decided to use the exact spans of mentions   pending on the similarity, there are two variations
for determining correctness.  The NP boundaries — entity based CEAF — CEAFe and a mention based
for the test data were pre-extracted from the hand-  CEAF — CEAFm.  Recall is the total similarity di-
tagged Treebank for annotation, and events trig-   vided by the number of mentions in K, and preci-
gered by verb phrases were tagged using the verbs   sion is the total similarity divided by the number of
themselves. This choice means that scores for the   mentions in R. Finally, BLANC uses a variation on
CoNLL-2012 coreference task are likely to be lower   the Rand index (Rand, 1971) suitable for evaluating
than for coreference evaluations based on MUC, or   coreference. There are a few other measures — one
ACE data, where an approximate match is often al-   being the ACE value, but since this is speciﬁc to a
lowed based on the speciﬁed head of the mentions.     restricted set of entities (ACE types), we did not con-
                                                         sider it.
4.5.1  Metrics
  As noted above, the choice of an evaluation met-   4.5.2  Ofﬁcial Evaluation Metric
ric for coreference has been a tricky issue and there      In order to determine the best performing system
does not appear to be any silver bullet that addresses   in the shared task, we needed to associate a single
all the concerns. Three metrics have been commonly
used for evaluating coreference performance over an      24The MUC corpora did not tag single mention entities.

                                        19
<a id="page-20"></a>

### PDF 第 20 页

number with each system.  This could have been   webpage. Of these, 16 groups from 6 countries sub-
one of the metrics above, or some combination of   mitted system outputs on the test set during the eval-
more than one of them. The choice was not sim-   uation week. 15 groups participated in at least one
ple, and after having consulted various researchers   language in the closed task, and only one group par-
in the ﬁeld, we came to a conclusion that each met-   ticipated solely in the open track. One participant
ric had its pros and cons and there is no silver bul-   (yang) did not submit a ﬁnal task paper. Tables 13
let. Therefore we settled on the MELA metric pro-   and 14 list the distribution of the participants by
posed by Denis and Baldridge (2009), which takes a   country and the participation by language and task
weighted average of three metrics: MUC, B-CUBED,   type.
and CEAF. The rationale for the combination is that
each of the three metrics represents a different, im-                      Country   Participants
portant dimension. The MUC measure is based on                                                                                                Brazil                1
links. The B-CUBED is based on mentions, and the                      China                8
CEAF is based on entities. We decided to use the en-                    Germany             3                                                                                                           Italy                 1
tity based CEAFe instead of mention based CEAFm.                         Switzerland           1
For a given end application, a weighted average of                USA                2
the three might be optimal, but since we don’t have
a particular end task in mind, we decided to use the             Table 13: Participation by country.
unweighted mean of the three metrics as the score
on which the winning system was judged. This still
leaves us with a score for each language. We wanted
to encourage researchers to run their systems on all                          Closed Open Combined
three languages. Therefore, we decided to compute                   English      15     1        16
the ﬁnal ofﬁcial score that would determine the win-                  ChineseArabic      137     31        148
ning submission as the average of the MELA metric
across all the three languages. We decided to give a      Table 14: Participation across languages and tracks.
MELA score of zero to every language that a partic-
ular group did not run its system on.
4.5.3  Scoring Metrics Implementation          6  Approaches
                                                      Tables 15 and 16 summarize the approaches taken  We used the same core scorer implementation25                                            by the participating systems along some importantthat was used for the SEMEVAL-2010 task, and                                                   dimensions. While referring to the participating sys-which implemented all the different metrics. There                                                    tems, as a convention, we will use the last name ofwere a couple of modiﬁcations done to this scorer                                                      the contact person from the participating team. It issince then.                                                  almost always the last name of the ﬁrst author of the
                                                system papers, or the ﬁrst name in case of conﬂicting
  1. Only  exact  matches were  considered  cor-                                                                last names (xinxin). The only exception is chunyang
      rect.   Previously,  for SEMEVAL-2010 non-                                             which is the ﬁrst name of the second author for that
     exact matches were judged partially correct                                                   system. For space and readability purposes, while
     with a 0.5 score if the heads were the same                                                         referring to the systems in the paper we will refer
    and the mention extent did not exceed the gold                                                         to the system by the primary contact name in italics
     mention.                                                       instead of using explicit citations.
  2. The modiﬁcations suggested by Cai and Strube     Most of the systems divided the problem into the
     (2010) have been incorporated in the scorer.      typical two phases — ﬁrst identifying the potential
                                                mentions in the text, and then linking the mentions
  Since there are differences in the version used for   to form coreference chains, or entities. Many sys-
CoNLL and the one available on the download site,   tems used rule-based approaches for mention detec-
and it is possible that the latter would be revised in    tion, though one, yang did use trained models, and
the future, we have archived the version of the scorer     li used a hybrid approach by adding mentions from
on the CoNLL-2012 task webpage.26                a trained model to the ones identiﬁed using rules.
                                                    All systems ran a post processing stage, after linking
5  Participants                                      potential mentions together, to delete the remaining
 A total of 41 different groups demonstrated in-   unlinked mentions. It was common for the systems
terest in the shared task by registering on the task   to represent the markables (mentions) internally in
   25http://www.lsi.upc.edu/∼esapena/downloads/index.php?id=3     27The participant did not submit a ﬁnal paper, so this infor-   26http://conll.bbn.com/download/scorer.v4.tar.gz               mation is based on an email correspondence.

                                        20
<a id="page-21"></a>

### PDF 第 21 页

(E);(C)
                T                 T
                                 –               the                                 of                                  of
  D D       D      D D D D D       D  + +       +      + + + + +                                    (2011) +     InTrain T T   T   T        20%15% T T T T T T    T T                                                                                                                                                                                                                                                                                                                                           dependency      (A)                                                                                                                                                                                                                tracks.a     (E);                                                                              2011                                                                                                                                      positive      223
                                                            61                                  templates(E)a        –                                                                              al.,                                                 33      groups –   (C)                                 –  40 71         both                                                                                                                                    used                                          ∼45                                                  and      and         34                                                               18–37     of                                       et                                                                                          and                                                                form                                                                or            templates(C)                           featureand                               the                                                                  feature        (E)                                 theyFeatures#  196197 28(C)                    Innegativerelations      Chang,           11      51                                    open                                                                                                                                    that
                                    set
    and                                                                                                                                                                                                                                  closed,                                                                                          starting     feature                                                              per                                                                                     feature    Arabic                                                                                                 the                                                                                                        but                                                                                                                                                                                                                                                                                                                                           represents                                                                                                                                                                                                                                                                                                  backward                                  selection                                 Soonand × × × × × × × ×                                                                in            induction                                 D                                                                                                                                          set,
                                       and                      a                                                                                            elimination               Backward
           +Selection                           forward         featuretemplates                                                                                                 Chinese                                                                                                                                           feature     forwardsystem                                                                                                                         CoNLL-2011                                                                                           English                                         for
                                       for                                         set                                                                               SameclassiﬁerGreedysieve                  whereasFeature   Latentfeature    Greedy(semi-automatic)                      Backward             Forwardfrom                                                                                                                                                                                                                                                                                                                                                                                                   participated                                                                                                                                                     2011system
Verb × ×  ×  ×  × × × × √ × × × ××al.,et                                                                                                                                                                                                                                       syntax,                                                                                                                                                        2011      system
                                                                                                               Lee                                                                  of                                                                                                                                                        al.,                         and        NR                                                     is                                                                                         are                                                                                                 the                                            and                                                                          of                                                                            et                                  in                                           to                                                in                                                                                                                                                named                     NP                                              are        to                                                                                              NPs                                                                                                                                                                                                 speech                                                                                           NPs         and                       in                                                 and                                                                                                                                                             entities                                                                                                                                                                                                                                entities              entities                                                                                                                                                           Arabic.                                                                                                                                                                          data.                                                    with                                                                                                                                                                         names                                                      English.                                                                                                                  Lee                                                                                                                                                                                                            training                                                                of                                 English                                              NEs        PN                             PNpleonastic                                                                                      types                                                                                   NR                                              prune              in                                                   in                                                                       all                                                                                                                                                                                                                                                                    version                                                                  and         in                                                                                                                                                                                                                                                              Chinese                                                                                                                                                                                                                                             classiﬁer                                                                                                                     English,                                                                                                                                                                                                                                             mentions                                                  where.                  to                                                                              name                                                                                                                name       name                                                                                                                                 part                                                                                                                                                                                                                                       selected                          a                 NEChinese                                              few                                                                                                                                                                                                                                  whether                                                            NPclassiﬁer.                                                                                                                                                                                                Classiﬁer                                                                                                                                                                                                                                            certain                                   Classiﬁer                                                                       and                                            for               a                                                                                                             and                                                                                                                                  mentions                         and                                                           and                                                                                    and     and                                                                                                        mentions                                                                                      and                                                                                                                                                                                                      speech                                                                                                                                                                                                                                                    mentions.                                                                                     Exclude                                                                               and                                                                and                                                                                                                        mention                                                                                                                              handled                                                                                                                                                                 using                                                                                                                                                                                                                                                                                                                                                                                        coordinating                                                                                                     and                                                                                                                                                                                                                                                                                                                                                                              second-level                                                                                                                                                                                                                                                                                                                                                                                                                                                                               representation                                              and                                                                                                                                                           Chinese                                                                  of                                                                                                                                                                                                            mentions                                                                                                                                                                                                                                                                    Modiﬁed                              languages;                                                                                                                 languages;                                                                         types                                              pronouns                                         what                                                     Exclude                                                                          Learning                                                      selected                                            in                                                                                           are                                                                        also                                 0.95).                                                                    Four                                                                                                                                                                                                                                                                                                                                                                                     development                                                                            are         all                                  all                                                                                                                                                                                              English,                                                                                                                                                                                                                                                              English                               English.                                                                                                                                                                                                                              Singleton                                                                                                                                                                                                                                                                                                                                      overlapping                                                                                                                                    part                                                                                           smaller                                                                                                                                                                         words           of                                                                                                                                                                                                                                                                                                                           classiﬁer      in                    and                       in                                                in                                                                         in                                                                                                 English.        in                                                                                                                                                                                                                                                    potential                                                                                                                                                     English                                                                  NPs                                                      are                                                                                                                                                             pronouns                                                                                                                                                                                                                                pronouns              pronouns                                                                                                                                                                                                                                       pronouns                                                                                                                                  embedded                                                                                                                                                                                                                  mentions                                                                                                            nations                                                                                                                                                                                                                                                                                                                                  represents                                                                                                                          they                                                                                                                                                                                                                           example                                                                                                      that                                                                                         and                                                                                                                                                                                                                                                                 pronouns                        in                                                                                 mentions                                 as                                                                                                      mention                                                     as          NE                                                                for                                                                 Arabic.                                                                                                                                                                     using                                              for                                                             Chinese.                                                                                                                                                                                                                                       grammar                                                                                              for                                                                 Prune                                                                   pronouns                                                                                                                                                                            Arabic.            PRP$                           PRP$                                             PRP$                                                                                               PRP$                                                                                                                                                 PRP$                               NE                  in                                                                                                and            all                                                                                                 C/O                                                                                                                          whenIdentiﬁcation               in                                                                                                                               types                                                                                                                                                          pronouns.                                                                                                                                                                                                                                            measure                                                                 well                                              use                                                                                                                                                                                                                         identify                                                                                              NP,                                                                                                          rules                                                                                                                              Copulas                                                                                                                                                             phrases,                                                                                                                                                                                                                                phrases,              phrases,                                    all                                                                                                                                                                                                                                       phrases,                                                                                                                                                                                                                                                                                                                           detection                                                                                 NP                       NE                                                                          and         and                    and                                  and                                                                       and                                                                                                             and                                                                                                                                                                                 markable                                                                      non-referential                                                                                                                                                                                                                                                                                                                                                   represents                                                                                                                                                                                                                                             selected                   QP                                 as                                                                                                                                                                     names                                                             probability                                                      to                                                               an                                                                     head.different                                  D                                                                                                        English.       noun  PRP                                                                              noun             PRP    phraseconsidered   nounnoun      a  PRPand           PRP                                                                                                   and         PRP                               Chinese;                          in                                                                                                                                                                                                                                                                                                          structureMarkable   All NP,inexclude(with NP,NPinterrogativeselectednon-referential NP,Chinese;itsameEightadjectivalﬁlteredpleonasticChinese.appropriately.All      Standardidentify  NP,ChineseAllaretrainedAllAllentitiesconsideredinsideChinesePNexcluding  Mention NP,                 column,  and

             for                                                                                                                                        data                                                                  for                            Lee                      Taskphrase          and                                     the
                                 a                                                                                        BART          using                                rules  and    of                the                                                                       learningspeciﬁc                the                                                                   overof                                  (Mallet)                                                                                                                                                                                                                                                                                                                                              structure)                                                                In                        and                                                     top                                                                                                                                                                                                                             classiﬁers                                                   approach                                                                                                                                                             Learning                  Perceptron                                                                                                                                    used                                                                                                                                                                                                                                                                               training                           linking,                                                                                  BART                                                                                                                                                                                                                                                                                     optimization.                                                                I.                                                                                                                                                                                              learning                                                Genre                                                                                                              2008))                                                                                                                                                                                                                                                                           classiﬁer                                                                                                                                                        rules                                         of                                                     (on                                                                                                                                                                       sieve          for                                                                                                                   language-speciﬁc                                                                                                                     learned                                                                                                                                                                                                                                                                                                                                                                      (adaptation                                                                         al.,                              Entropy                                Sieve                                                                                                         SVMFramework                                                                                                                                 Part                                                                                                                       tree              by                                                        pruning                                                                                                                       based                                         deterministic                                     et                                                                                                                         Regression                   multigraphwhere                                                  are                                                                      data                                                                                                                                                           speciﬁcgenre.bc      —                                                                                                                                                                                Structure                Structure                                                                             pruning;                                                                                                                                                                                                                                       systems                                —                                                                                                                                                                                                        2011’s                                                                                    and
                                                                     and                                                                                                    al.                                                                representsLearning      Latent      LIBLINEARMaximumanaphoricity  Hybridfollowedheuristiclanguage-independentbasedmodels     Logistic(LIBLINEAR)         Directedrepresentationweightstraining(Versley Latentmodiﬁcationmulti-objectiveDomainnwMemory(TIMBL)    MaxEnt   C4.5      Decisiondeterministic           Rule-basedet       Structural  MaxEnt                                                                                                                                                                                                                                  proﬁlesthethatT                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        Bj¨orkelund.Syntax P P   P   D   D  P P P P P P  P P PPP                                                                                                                                                                                                           column
                                                                                                                                                                                                 system               Anders
  E E   E   E        E E E           E                                                                                                                                                                          Train    C,   C,      C,      C,   C  C  C,   C,  C, C C  C  C C,                   represents                                                                                                                                                              withLanguages  A,   A,      A,      A,      A,    E,  A,   A,  A, E,  E,    E, E E,EA,  Pthe
                                 aIn
        O                   OTrack C C      C,   C   C  C C C C C, C  C C CCO             Participating                                                                                                                                                                                                                                       column,                                                                                                 15:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 aCommunicationParticipant        fernandes                bj¨orkelund             chen                         stamborg                            martschat         chang         uryupina          zhekova  li   yuan  xu               chunyang      yang27 xinxinshouxiong     TableSyntaxrepresentation.


                                     21
<a id="page-22"></a>

### PDF 第 22 页

used.
         for                        —
                                                           for                                                                                                                           window                       other                                                                          mentionsentences.     tree   ﬁrstdisallow         by                                                                 Best-ﬁrst                                                          rules                                                              strategy                                                                averaged    greedy                      each                                       of                         mentions                      and                                                                                                                                                                    sentence                                                                                                                                         (Arabic)                        and by                     non-pronominal             with                                                                                                                                                                                                                                                                                    closest-ﬁrst           document     closest—                 7     seta                                     followed                                                                                  5/10                                                                                                                                                                       linking                                                                     use                                                                                                                              antecedents.  and
                     a                       ﬁrst,                                                of                                   decoding                                                                                                  using                                                                             followed                                         of         scoring   ﬁrstcluster-mention;                                                         pronouns                                                                     and                                                                                                                  anaphoror                                                                pronouns                                                            2001                                                                         the               iii)                      for                                                                                               type                     closest                                                                                                                                         Chinese)                        for                                                                 each                                                                                              candidate                                                                        the                                                                                                        Chinese.           and                                                                         and                                                            Soon                                          processed                     of       pronominal                                                 for                                                                                                                                                                                                     correction                                                                                                 clustering                                                on         maximum                              in                                       for                                    for                                       or                                                                                                                                                outside                    pre-clusters,                              as                     Pronoun                                                                        clustering                                                                 pair    the                                                                                                                                                          (English,                                                                                              PP-NP         ii)                                                                                                                       based                a10                               mentions                                      closest-ﬁrst         number                                                                                                                                                                                         sentences                                     mentions                                                                             Spectral                                      by     ﬁnds        —                                  of                                                                                                                                                                                                                          generated               ﬁrst,                                                                                                                                          pruningfor           pronoun                                                                                                                         classiﬁersclassiﬁer       ranking create                   other                   noun                             reduce          to     by                   models                                                                                                                                                                                                                                                                respectively                                                                                      Closest-ﬁrst            Best           for                     to                                                                                                                           NP-NP                                            pruning                                                        English                                                                                                                                                      followed      i)                                                                                               link            predictor       —                                                                                                                                                            method           were                                                                                        otherwise.English;                                                                              Single       without usedwindowa                                                                                                                                                                    mentions                                                                                                 separate                             for                                                                          and               ﬁrstproper   —                                                                                                                                                                                                               singleton                                                                                                                                                             followed                      candidates                                based                                                                                                                                                     Chinese        latent                                                                                              NP-NP                                                                         Chinese                                                                                                        English.                                                                           model                                                                                            with                                           Arabic                                                                                                                  phraseswithin                                                                                                       NP-NP                                                                                                                                                                                                                       Different                                                                and                                            pronouns                                                                                otherwise.                                                                                                                                                                                                  examples           andnesting;                                       for                                  itclustering                               for      strategy;                                                                                        clustering                                                                                                                                                                             PRP-PRP  clustering system                                                            pair                      and                                                                                                                                          linking                           resolversbest                                                                                                 clustering                 possible                                                              and                and                                                  learning                                                 link                                                                                                                  deﬁnite               constrained            and                                                                                                  EnglishDecoding AamongStackedpronounstransitivenouns     Chineseclusteringbest-ﬁrstGreedyclustering   Bestmentions       mention AllprecedingBest-ﬁrstDeterministicAll-pairNE-NEin         Pre-clusters,clustering.NP-PRP,  Best-ﬁrst   coreference                                                                                                                                                                                                  negative                                                                                                                                     system                             system
                                    the                                                                                                             2011     and           and             heuristic and              gold                                            and  and   a             and
                                                                                                             al.,                                                                                         2011                                                                                                                2011             by        of           proposed                                                      et                                          is                       ofwhere                                                                                         al.,                                                                                                                al.,                                       not                                           Englishused
                                                        et                                                                                 Lee                           anaphor                                                     anaphor                                                                                                      anaphor                                                                                                                  anaphor                                                                                                                                                                                   anaphor                                                                                                                                                                                                  positive                                       are                                                      of                                                         union        methodrulesof                                                  followed                                                                  Lee     PRP-PRP                                                                                    Lee
           a                       2001)                                          2001)                                                                             2001)                                                                                      2001)                                                                                                                                    2001)                                       not      set                                                            Forsentences et                                                                         way                       inMentions  a                                                                        and                                                                                                                                                                                              version     theExamples   sievethe          between                   between                                          between     between       10                         between                             in(Soon,in(Soon,                  examples.of       in(Soon,               in(Soon,   data         other        in(Soon,     approach           using                                       and                                                           NP-PRP         on                                                                                          mentionsmentions.                                                                                                                                                                                              Modiﬁed                                                                                                                             window                                                                                 trainingTraining   usingpairs                sieve                     a            NP-NP,                               examplesantecedent                examplesantecedent                                                                                                                     examplesantecedentexamplesantecedent                negative                                                                                                                                                                                                            examplesantecedent        recall                              the             pronoun                                                            and                                                                                                                                                                          focuses               mention              of  precedingpredicted     high                     is                                        pairNegative                        Negativeclosest       Rule-based  Negativeclosest   part AllandﬁrstconsideredNegativeclosestNegativeclosest                    Chinese                             Negativeclosest         This                  ——  positive           each
                    on                              for                    II.               between                                                              for           generate
                                                                                                 Part   to         arcs                                                                      trained                                                                                                                                                  whereas                                                                                                                                                                                     determine                        —    aim                                                                                                                                                                                               separate                              are                                        to   an                                                                          used                 directed                     2001)              2001)                  2001)       2001)   2001)          usedis                    2001)     with                                        is                 models                               proﬁles                                                                      Weights             Create                         (Soon,                 (Soon,                     (Soon,         (Soon,    (Soon,                         pair      (Soon,Examples                                                                                                                                                                                            sentences           mentions                   5             2011).                                                                                                                                                        sentences                                            system
                                          of
         al.,                                  ofTraining   likelyet           Antecedent                            Antecedent                                    Antecedent               Antecedent       Antecedent                                                                    Pre-cluster           Antecedent
                                                                                                                             window         (Lee                                                                                                      WindowPositive    Identifyin       Closest                   Closest                         Closest          Closest     Closest                                                          Closest                                                                                                                                                                                                                                                                                                                           Participating
                                                                         16:Participant        fernandes                bj¨orkelund    chen      stamborg          martschat       chang            uryupina     zhekova liyuan xu     chunyang   yang      xinxin shouxiong                                                                                                                         Table



                         22
<a id="page-23"></a>

### PDF 第 23 页

terms of individual parse tree NP constituent spans.    that 先生(sir) and 女士(lady) often suggest gender
Some systems consider only mention-speciﬁc at-   information. bo and martschat used plurality mark-
tributes while performing the clustering, but the re-   ers 们to identify plurals. For example, 同学(stu-
cent trend seems to indicate a shared attribute model,   dent) is singular and 同学们(students) is plural. bo
where the attributes of an entity are determined col-   also uses a heuristic that if the word 和(and) appears
lectively by heuristically merging the attribute types   in the middle of a mention M, and the two parts sep-
and values of its constituent mentions. For example,   arated by 和are sub-mentions of M, then mention M
if a mention marked singular is clustered with an-    is considered to be plural. Other words which have
other entity marked plural, then the collective num-   the similar meaning of 和, such as 同, 与and 跟,
ber for the entity is assigned to be {singular, plu-   are also considered. uryupina used the rich part of
ral}.  Various types of trained models were used   speech tags to classify pronouns into subtype, per-
for predicting coreference.  For a learning-based   son number and gender. Chinese and Arabic do not
system, generation of positive and negative exam-   have deﬁnite noun phrase markers like the in En-
ples is very important.  The participating systems    glish.  In contrast to English there is no strict en-
used a range of sentence windows surrounding the   forcement of using deﬁnite noun phrases when re-
anaphor in generating these examples.  In the sys-                                                         ferring to an antecedent in Chinese. Both 这次演
tems that used trained models, many systems used                                      说(the talk) and 演说(talk) can corefer with thethe approach described in Soon et al. (2001) for se-                                                    antecedent 克林顿在河内大选的演说(Clinton’slecting the positive and negative training examples,                                                           talk during Hanoi election). This makes it very difﬁ-while others used some of the alternative approaches                                                           cult to distinguish generic expressions from referen-that have been introduced in the literature more re-                                                                     tial ones. martschat checks whether the phrase startscently. Following on the success of rule-based link-                                                  with a deﬁnite/demonstrative indicator (e.g., 这(this)ing model in the CoNLL-2011 shared task, many                                                     or 那(that)) in order to identify demonstrative andsystems used a completely rule-based linking model,                                                    deﬁnite noun phrases. For Arabic, uryupina consid-or used it as a initializing, or intermediate step in a                                                         ers as deﬁnite all mentions with deﬁnite head nounslearning based system. A hybrid approach seems                                                 (preﬁxed with “Al”) and all the idafa constructs withto be a central theme of many high scoring sys-                                                  a deﬁnite modiﬁer. chang uses training data to iden-tems. Also, taking cue from last year’s systems, al-                                                                tify inappropriate mention boundaries.  They per-most all systems trained pleonastic it classiﬁers, and                                             form a relaxed matching between predicted men-used speaker-based constraints/features for the con-                                                         tions and gold mentions ignoring punctuation marksversation genre. Many systems used the predicted                                              and mentions that start with one of the following:Arabic parts of speech that were mapped-down to                                                   adverb, verb, determiner, and cardinal number. InPenn-style parts of speech, but stamborg used some                                                    another extreme, xiong translated Chinese and Ara-heuristics to convert them back to the complex part                                                      bic to English, and ran an English system and pro-of speech type, using more frequent mapping, to                                                        jected mentions back to the source languages. Un-get better performance for Arabic. The fernandes                                                          fortunately, it did not work quite well by itself. Onesystem uses feature templates deﬁned on mention                                                       issue that they faced was that many instances of pro-pairs. bj¨orkelund mentions that disallowing transi-                                              nouns did not have a corresponding mention in thetive closures gave performance improvement of 0.6                                                   source language (since we do not consider mentionsand 0.4 respectively for English and Chinese/Arabic.                                              formed by dropped subjects/objects). Nevertheless,bj¨orkelund also mentions seeing a considerable in-                                                   using this in addition to performing coreference res-crease in performance after adding features that cor-                                                       olution in these languages could be useful. Similarrespond to the Shortest Edit Script (Myers, 1986)                                                         to last year, most participants appear not to have fo-between surface forms and unvocalised Buckwal-                                                 cused much on eventive coreference, those corefer-ter forms, respectively.  These could be better at                                                ence chains that build off verbs in the data. This usu-capturing the differences in gender and number sig-                                                           ally means that nominal mentions that should havenaled by certain morphemes than hand-crafted rules.                                                     linked to the eventive verb were instead linked inchen built upon the sieve architecture proposed in                                                  with some other entity, or remained unlinked. Par-Raghunathan et al. (2010) and added one more sieve                                                           ticipants may have chosen not to focus on events be-— head match — for Chinese and modiﬁed two                                                  cause they pose unique challenges while making upsieves. Some participants tried to incorporate pecu-                                                  only a small portion of the data (Roughly 90% ofliarities of the corpus in their systems. For example,                                                mentions in the data are NPs and pronouns). Many ofmartschat excluded adjectival nation names. Unlike                                                      the trained systems were also able to improve theirEnglish, and especially in absence of an external re-                                                performance by using feature selection, the detailssource,  it is hard to make a gender distinction in                                                      varied depending on the example selection strategyArabic and Chinese. martschat used the information                                              and the classiﬁer used.


                                        23
<a id="page-24"></a>

### PDF 第 24 页

7  Results                                          the best with a score of 62.24. This is then followed
  In this section we will take a look at the perfor-   by yuan (60.69), and then bj¨orkelund (59.97) and
mance overview of various systems and then look   xu (59.22).  It is interesting to note that the scores
at the performance for each language in various set-   for the top performing systems for both English and
tings separately. For the ofﬁcial test, beyond the raw   Chinese are very close. For all we know, this is just
source text, coreference systems were provided only   a coincidence. Also, for both English and Chinese,
with the predictions for the other annotation layers   the top performing system is almost 2 points higher
(parses, semantic roles, word senses, and named en-   than the second best system.
tities). A high-level summary of the results for the    On the Arabic language front, once again, fernan-
systems on the primary evaluation for both open and   des has the highest score of 54.22, followed closely
closed tracks is shown in Table 17. The scores un-   by bj¨orkelund (53.55) and then uryupina (50.41)
der the columns for each language are the average                                                    Since the majority of mentions in all the threeof MUC, BCUBED and CEAFe for that language. The                                                  languages are noun phrases or pronouns, the accu-column Ofﬁcial Score is the average of those per-                                                    racy with which these are predicted in the parse treeslanguage averages, but only for the closed track. If a                                                  should directly bear on the coreference scores. Sinceparticipant did not participate in all three languages,                                                pronouns are a closed class and single words, thethen they got a score of zero for the languages that                                            main focus falls on the accuracy of the noun phrases.were not attempted. The systems are sorted in de-                                      By no means is the accuracy of noun phrases thescending order of this ﬁnal Ofﬁcial Score. The last                                                  only factor determining the overall coreference ac-two columns indicate whether the systems used only                                                       curacy, but it cannot be ignored either. It can be ob-the training or both training and development for                                                   served that the coreference scores for the three lan-the ﬁnal submissions. Most top performing systems                                                guages are in the same trend as the noun phrase ac-used both training and development data for training                                                      curacies for those languages as seen in Table 6. Re-the ﬁnal system. Note that all the results reported                                                             call that in case of both Chinese and Arabic, therehere still used the same, predicted information for                                                      are roughly 11% instances of dropped pronouns thatall input layers.                                              were not considered as part of the evaluation. The   It can be seen that fernandes got the highest com-                                                performance for Chinee and Arabic would decreasebined score (58.69) across all three languages and                                            somewhat if these were considered in the set of goldmetrics. While scores for each individual language                                                mentions (and entities).are lower than the ﬁgures cited for other corpora, it
                                                      Tables 18 and 19 show similar information for theis as expected, given that the task here includes pre-
                                             two supplementary tasks — one given gold mentiondicting the underlying mentions and mention bound-
                                                 boundaries (GB) and one given correct, gold men-aries, the insistence on exact match, and given that
the relatively easier appositive coreference cases are   tions (GM). We have however, kept the same rela-
                                                             tive ordering of the system participants as in Tablenot included in this measure. The combined score
                                            17 for ease of reading. Looking at Table 18 care-across all languages is purely for ranking purposes,
                                                                fully, we can see that for English and Arabic the rel-and does not really tell much about each individual
                                                          ative ranking of the systems remain almost the same,language. Owing to the ordering based on ofﬁcial
                                                   except for a few outliers: chang performs the bestscore, not all the best performing systems for a par-
                                                   given gold mentions — by almost 7 points over theticular language are in sequential order. Therefore,
                                                    next best performing system. In the case of Chinese,for easier reading, the scores of the top ranking sys-
                                               chen performs almost 6 points better than the ofﬁcialtem are in bold red, and the top four systems are
                                                performance given gold boundaries, and another 9underlined in the table.
                                                      points given gold mentions and almost 8 points bet-  Looking at the the English performance, we can
                                                               ter than the next best system using gold mentions.see that fernandes gets the best average across the
                                  We will look at more details in the following sec-three selected metrics (MUC, BCUBED and CEAFe).
                                                            tions.The next best system is martschat (61.31) followed
very closely by bj¨orkelund (61.24) and then chang    As mentioned earlier in Section 4.2 we conducted
(60.18). The performance differences between the   some supplementary evaluations. These can be cat-
better-scoring systems were not large, with only   egorized by a combination of two parameters. One
about three points separating the top four systems,   of which applies to both training and test set, and
and only six out of a total of sixteen systems getting   one can only apply to the test set. The two parame-
a score lower than 58 points which was the highest    ters are: i) Syntax and ii) Mention Quality. Syntax
performing score in CoNLL-2011.28                can take two values:  i) predicted (PS), or ii) gold
  In case of Chinese, it is seen that chen performs   (GS), and can be applicable during either training or
                                                                  test; and, the mention quality can be of three values:
  28More precise comparison later in Section 8.                     i) No boundaries (NB), ii) Gold mention boundaries

                                        24
<a id="page-25"></a>

### PDF 第 25 页

Participant        Open                   Closed           Ofﬁcial  Final model

                                  English  Chinese  Arabic  English  Chinese  Arabic   Score  Train  Dev
                      fernandes                               63.37    58.49   54.22    58.69  √  √
                        bj¨orkelund                              61.24    59.97   53.55    58.25  √  √
                     chen                   63.53            59.69    62.24   47.13    56.35  √                     stamborg                               59.36    56.85   49.43    55.21  √  √×
                     uryupina                               56.12    53.87   50.41    53.47  √  √
                     zhekova                                48.70    44.53   40.57    44.60  √  √
                                     li                                       45.85    46.27   33.53    41.88  √  √
                    yuan                  61.02            58.68    60.69            39.79  √  √
                    xu                                      57.49    59.22            38.90  √                      martschat                              61.31    53.15            38.15  √  ×                    chunyang                               59.24    51.83            37.02   –   ×–
                    yang                                    55.29                     18.43  √                    chang                                  60.18    45.71            35.30  √  ×                       xinxin                                  48.77    51.76            33.51  √  √×
                     shou                                    58.25                     19.42  √                      xiong         59.23    44.35   44.37                                0.00  √  √×

            Table 17: Performance on primary open and closed tracks using all predicted information.





                        Participant        Open                   Closed           Suppl.  Final model

                                   English  Chinese  Arabic  English  Chinese  Arabic  Score  Train  Dev
                       fernandes                               63.16    61.48   53.90   59.51  √  √
                         bj¨orkelund                              60.75    62.76   53.50   59.00  √  √
                     chen                   70.00            60.33    68.55   47.27   58.72  √                     stamborg                               57.35    54.30   49.59   53.75  √  √×
                     zhekova                                49.30    44.93   40.24   44.82  √  √
                                      li                                       43.04    43.28   31.46   39.26  √  √
                    yuan                                   59.50    64.42           41.31  √  √
                    xu                                      56.47    64.08           40.18  √                     chang                                  60.89                    20.30  √  √×

Table 18: Performance on supplementary open and closed tracks using all predicted information, given gold mention
boundaries.





                        Participant        Open                   Closed           Suppl.  Final model

                                   English  Chinese  Arabic  English  Chinese  Arabic  Score  Train  Dev
                       fernandes                               69.35    66.36   63.49   66.40  √  √
                         bj¨orkelund                              68.20    69.92   59.14   65.75  √  √
                     chen                   78.98            70.46    77.77   52.26   66.83  √                     stamborg                               68.66    66.97   53.35   62.99  √  √×
                     zhekova                                59.06    51.44   55.72   55.41  √  √
                                      li                                       51.40    59.93   40.62   50.65  √  √
                    yuan                                   69.88    76.05           48.64  √  √
                    xu                                      63.46    69.79           44.42  √                     chang                                  77.22                    25.74  √  √×

Table 19: Performance on supplementary open and closed tracks using all predicted information, given gold mentions.





                                        25
<a id="page-26"></a>

### PDF 第 26 页

80

                 70

                 60

                 50
                       fernandes                    bjorklund
                 40
                                                                       chen                   stamborg

                 30

                 20                                    Arabic           Chinese            English


                 80

                 70

                 60

                                               li
                 50
                                                xu                     yuan                     chang
                 40

                 30

                 20
  Testing
   Mention Quality
    No Boundaries
    Gold Boundaries
    Gold Mention

  Figure 4: Performance for eight participating systems for the three languages, across the three mention qualities.


(GB) and iii) Gold mentions (GM), and can only be   gain for Chinese.
applicable during testing (since this information is                                                     Figure 5 is a box and whiskers plot of the per-not optional during training, as is the case with us-                                               formance for all the systems for each language anding gold or predicted syntax). There are a total of                                                        variations — NB, GB, and GM. The circle in the cen-twelve combinations that we can form of using these                                                               ter indicates the mean of the performances. The hor-parameters. Out of these, we thought six were par-                                                         izontal line in between the box indicates the median,ticularly interesting. This is the product of the three                                              and the bottom and top of the boxes indicate the ﬁrstcases of mention quality — NB, GB and GM, and two                                              and third quartiles respectively, with the whiskers in-cases of syntax – GS and PS used during testing.                                                       dicating the highest and lowest performance on that
                                                           task.  It can be easily seen that the English systems  Figure 4 shows a performance plot for eight par-                                                have the least divergence, with the divergence largeticipating systems that attempted both the supple-                                                         for the GM case probably owing to chang. This ismentary tasks — GB and GM in addition to the main                                            somewhat expected as this is the second year for theNB for at least one of the three languages. These                                                   English task, and so it does show a more mature andare all in the closed setting. At the bottom of the                                                         stable performance. On the other hand, both Chineseplot you can see dots that indicate what test condi-                                              and Arabic plots show much more divergence, withtion to which a particular point refers. In most cases,                                                      the Chinese and Arabic GB case showing the highestfor the hardest task — NB — the English and Chi-                                                     divergence. Also, except for Chinese GM condition,nese performances track quite close to each other.                                                       there is some skewness in the score distribution oneWhen provided with gold mention boundaries (GB),                                        way or the other.systems, chen, xu and yuan do signiﬁcantly better
in Chinese.  There is almost no positive effect on    Some participants ran their systems on six of
the English performance across the board. In fact,   the twelve possible combinations for all three lan-
performance of the stamborg and li systems drops   guages.  Figure 6 shows a plot for these tree par-
noticeably. There is also a drop in performance for    ticipants — fernandes, bj¨orkelund, and chen. As in
the bj¨orkelund system, but the difference is probably   Figure 4, the dots at the bottom help identify which
not signiﬁcant.  Finally, when provided with gold   particular combination of parameters the point on
mentions, the performance of all systems increases   the plot represents.  In addition to the three test
across all languages, with chang showing the high-   conditions related to mention quality, we now also
est gain for English, and chen showing the highest   have two more test conditions relating to the syntax.

                                        26
<a id="page-27"></a>

### PDF 第 27 页

We can see that the fernandes and bj¨orkelund, sys-   improvement when gold parse is used for training,
tem performance tracks very close to each other. In   only when gold mentions are available during test-
other words, using gold standard parses during test-   ing.
ing does not show much beneﬁt in those cases. In    One point to note  is that we cannot compare
case of chen, however, using gold parses shows a   these results to the ones obtained in the SEMEVAL-
signiﬁcant jump in scores for the NB condition.  It   2010 coreference task which used a small portion of
seems that somehow, chen makes much better use   OntoNotes data because it was only using nominal
of the gold parses. In fact, the performance is very    entities, and had heuristically added singleton men-
close to the one with the GB condition. It is not clear    tions29.
what this system is doing differently that makes this
possible.  Adding more information,  i.e., the GM                                                             29The documentation that comes with the SEMEVAL data
condition, improves the performance by almost the   package from LDC (LDC2011T01)  states:  “Only nominal
same delta as going from NB to GB.                     mentions and identical (IDENT) types were taken from the
                                                        OntoNotes coreference annotation, thus excluding coreference
   Finally, Figure 7 shows the plot for one system —    relations with verbs and appositives. Since OntoNotes is only
                                                              annotated with multi-mention entities, singleton referential ele-bj¨orkelund — that was ran on ten of the twelve dif-                                                        ments were identiﬁed heuristically: all NPs and possessive de-
ferent settings. As usual the dots at the bottom help    terminers were annotated as singletons excluding those func-
identify the conditions for a point on the plot. Now,    tioning as appositives or as pre-modiﬁers but for NPs in the pos-
there is a condition related to the quality of syntax    sessive case.  In coordinated NPs, single constituents as well
                                                                 as the entire NPs were considered to be mentions. There is noduring training as well.  For some reasons, using                                                                        reliable heuristic to automatically detect English expletive pro-
gold syntax hurts performance — though slightly —    nouns, thus they were (although inaccurately) also annotated as
in the NB and GB settings. Chinese does show some    singletons.”




           100




            80




            60




            40




            20




             0
                                    (NB)         (GB)         (GM)         (NB)         (GB)         (GM)         (NB)         (GB)         (GM)
                                                     Arabic              Arabic              Arabic                Chinese                Chinese                Chinese                English                English                English

  Figure 5: A box and whiskers plot of the performance for the three languages across the three mention qualities.


                                        27
<a id="page-28"></a>

### PDF 第 28 页

Arabic           Chinese            English


                       100

                        90

                        80

                        70

                        60

                        50

                        40      fernandes                 bjorklund                 chen

                        30

                        20

                        10

                       0          Testing
          Syntax
            Gold
             Predicted

          Mention Quality
          No Boundaries
            Gold Boundaries
            Gold Mention


               Figure 6: Performance of fernandes, bj¨orkelund and chen over six different settings.


  In the following sections we will look at the re-   they not get credit for the singleton entities that they
sults for the three languages, in various settings in   incorrectly removed from the data, but they will be
more detail. It might help to describe the format of   penalized for the ones that they accidentally linked
the tables ﬁrst. Given that our choice of the ofﬁcial   with another mention. What this number does in-
metric was somewhat arbitrary, it is also useful to   dicate is the ceiling on recall that a system would
look at the individual metrics. The tables are simi-   have got in absence of being penalized for making
lar in structure to Table 20. Each table provides re-   mistakes in coreference resolution. The tables are
sults across multiple dimensions. For completeness,   sub-divided into several logical horizontal sections
the tables include the raw precision and recall scores   separated by two horizontal lines. There can be a
from which the F-scores were derived. Each table    total of 12 sections, each categorized by a combi-
shows the scores for a particular system for the task   nation of two parse quality features GS and PS for
of mention detection and coreference resolution sep-   each training and test set and three variations on the
arately. The tables also include two additional scores   mention qualities — NB, GB, and GM, as described
(BLANC and CEAFm) that did not factor into the of-    earlier. Just like we used the dots below the graphs
ﬁcial score. Useful further analysis may be possible    earlier to indicate the parameters that were chosen
based on these results beyond the preliminary results   for a particular point on the plot, we use small black
presented here. As you recall, OntoNotes does not   squares in the tables after the participant name, to
contain any singleton mentions. Owing to this pecu-   indicate the conditions chosen for the results on that
liar nature of the data, the mention detection scores   particular row. Since there are many rows to each
cannot be interpreted independently of the corefer-    table, in order to facilitate ﬁnding which number we
ence resolution scores. In this scenario, a mention   are referring to, we have added a ID column which
is effectively an anaphoric mention that has at least   uses letters e, c, and a to refer to the three languages
one other mention coreferent with  it in the docu- — English, Chinese and Arabic. This is followed by
ment. Most systems removed singletons from the   a decimal number, in which the number before the
response as a post-processing step, so not only will   decimal identiﬁes the logical block within the table


                                        28
<a id="page-29"></a>

### PDF 第 29 页

Arabic           Chinese            English
                                        80




                                        70




                                        60




                                        50




                                        40

                                Training
                             Syntax
                                  Gold
                                     Predicted

                               Testing
                             Syntax
                                  Gold
                                     Predicted

                             Mention Quality
                            No Boundaries
                                  Gold Boundaries
                                  Gold Mention

                         Figure 7: Performance of bj¨orkelund over ten different settings.


that share the same experiment parameters, and the   used machine learned classiﬁers for mention detec-
one after the decimal indicates the index of a par-    tion. This could be possible because any classiﬁer
ticular system in that block. Systems are sorted by    that is trained will not normally contain singleton
the ofﬁcial score within each block.  All the sys-   mentions (as none have been annotated in the data)
tems with NB setting are listed ﬁrst, followed by GB,   unless one explicitly adds them to the set of train-
followed by GM. One participant (bj¨orkelund) ran   ing examples (which is not mentioned in any of the
more variations than we had originally planned, but   respective system papers). A hybrid rule-based and
since it falls under the general permutation and com-   machine learned model (fernandes) performed the
bination of the settings that we were considering, it    best. Apart from some local differences, the rank-
makes sense to list those results here as well.          ing for all the systems is roughly the same irrespec-
                                                             tive of which metric is chosen. The CEAFe mea-
7.1  English Closed                                 sure seems to penalize systems more harshly than
  Table 20 shows the performance for the English   the other measures.  If the CEAFe measure does in-
language in greater detail.                             dicate the accuracy of entities in the response, this
                                                    suggests that fernandes is doing better on getting co-Ofﬁcial Setting  Recall is quite important in the                                                     herent entities than any other system.mention detection stage because the full coreference
system has no way to recover if the mention de-  Gold Mention Boundaries  In this case, all possi-
tection stage misses a potentially anaphoric men-   ble mention boundaries are provided to the system.
tion. The linking stage indirectly impacts the ﬁnal   This is very similar to what annotators see when
mention detection accuracy. After a complete pass   they annotate the documents. One difﬁculty with
through the system some correct mentions could re-    this supplementary evaluation is that these bound-
main unlinked with any other mentions and would   aries alone provide only very partial information.
be deleted thereby lowering recall. Most systems   For the roughly 10 to 20% of mentions that the auto-
tend to get a close balance between recall and preci-   matic parser did not correctly identify, while the sys-
sion for the mention detection task. A few systems   tems knew the correct boundaries, they had no struc-
had a considerable gap between the ﬁnal mention    tural syntactic or semantic information, and they
detection recall and precision (fernandes, xu, yang,   also had to further approximate the already heuris-
li and xinxin).  It is not clear why this might be the    tic head word identiﬁcation. This incomplete data
case. One commonality between the ones that had   complicates the systems’ task and also complicates
a much higher precision than recall was that they   interpretation of the results. While most systems did

                                        29
<a id="page-30"></a>

### PDF 第 30 页

slightly better here in terms of raw scores, the per-   7.2  Chinese Closed
formance was not much different from the ofﬁcial                                                     Table 21 shows the performance for the Chinese
task, indicating that mention boundary errors result-                                                 language in greater detail.
ing from problems in parsing do not contribute sig-
niﬁcantly to the ﬁnal output.30                         7.2.1  Ofﬁcial Setting
                                                        In this case, it turns out that chen does about 2Gold Mentions  Another supplementary condition                                                      points better than the next best system across all thethat we explored was if the systems were supplied                                                       metrics. We know that this system had some morewith the manually-annotated spans for all and only                                                  Chinese-speciﬁc improvements.   It is strange thatthose mentions that did participate in the gold stan-                                                  fernandes has a much lower mention recall with adard coreference chains. This supplies signiﬁcantly                                        much higher precision as compared to chen. As farmore information than the previous case, where ex-                                                     as the system descriptions go, both systems seem toact spans were supplied for all NPs, since the gold                                                have used the same set of mentions — except formentions will also include verb headwords that are                                               chen including QP phrases and not considering inter-linked to event NPs, and will not include singleton                                                       rogative pronouns. One thing we found about chenmentions, which do not end up as part of any chain.                                           was that they dealt with nested NPs differently inThe latter constraint makes this test seem artiﬁcial,                                                   case of the NW genre to achieve some performancesince it directly reveals part of what the systems are                                                improvement.  This unfortunately seems to be ad-designed to determine, but it still has some value in                                                     dressing a quirk in the Chinese newswire data owingquantifying the impact that mention detection and                                                         to a possible data inconsistency in the release.anaphoricity determination has on the overall task
and what the results are if they are perfectly known.   7.2.2  Gold Mention Boundaries
The results show that performance does go up signif-
                                                   Unlike English, just the addition of gold mentionicantly, indicating that it is markedly easier for the
                                                 boundaries improves the performance of almost allsystems to generate better entities given gold men-
                                                 systems signiﬁcantly.  The delta improvement fortions. Although, ideally, one would expect a perfect
                                                  fernandes turns out to be small, but it does gain onmention detection score, it is the case that many of
                                                      the mention recall as compared to the NB case.  Itthe systems did not get a 100% recall. This could
                                                                  is not clear why this might be the case. One ex-possibly be owing to unlinked singletons that were
                                                     planation could be that the parser performance forremoved in post-processing. chang along with fer-
                                                        constituents that represent mentions — primarily NPnandes are the only systems that got a perfect 100%
                                               might be signiﬁcantly worse than that for English.recall. The reason is most likely because they had
                                           The mention recall of all the systems is boosted bya hard constraint to link all mentions with at least
                                                  roughly 10%.one other mention. chang (77.22 [e7.00]) stands out
in that it has a 7 point lead on the next best sys-                                                       7.2.3  Gold Mentions
tem in this category.  This indicates that the link-
ing algorithm for this system is signiﬁcantly superior     Providing gold mention information further sig-
than the other systems — especially since the perfor-   niﬁcantly boosts all systems. More so is the case
mance of the only other system that gets 100% men-   with chen [e8.00] which gains another 9 points over
tion score (fernandes) is much lower (69.35 [e7.03])   the gold mention boundary condition in spite of the
                                                           fact that they don’t have a perfect recall. On the
Gold Test Parses  Looking at Table 20 it can be   other hand, fernandes gets a perfect mention recall
seen that there is a slight increase (∼1 point) in per-   and precision, but ends up getting a 11 point lower
formance across all the systems when gold parses   performance [c8.05] than chen.  Another thing to
across all settings — NB, GB, and GM. In the case   note is that for the CEAFe metric, the incremental
of bj¨orkelund for the NB setting, the overall perfor-   drop in performance from the best to the next best
mance improves by a percent when using gold test   and so on, is substantial, with a difference of 17
parse during testing (61.24 [e0.02] vs 62.23 [e1.02]),   points between chen and fernandes.  It does seem
but strangely if gold parses are used during train-    that the chen and yuan algorithm for linking is much
ing as well, the performance is slightly lower (61.71   better than the others.
[e3.00]), although this difference is probably not sta-
tistically signiﬁcant.                                   7.2.4  Gold Test Parses
                                       When provided with gold parses for the test set,
                                                       there is a substantial increase in performance for the
   30It would be interesting to measure the overlap between the                                      NB condition – numerically more so than in case ofentity clusters for these two cases, to see whether there was
any substantial difference in the mention chains, besides the ex-   English. The degree of improvement decreases for
pected differences in boundaries for individual mentions.        the GB and GM conditions.

                                        30
<a id="page-31"></a>

### PDF 第 31 页

3   Ofﬁcial          63.3761.3161.2460.1859.6959.3659.2458.6858.2557.4956.1255.2948.7748.7045.85 64.5362.8262.2361.4748.91 60.36 61.71 63.1660.8960.7560.3359.5057.3556.4749.3043.04 63.6761.5461.1857.5449.45 61.23 77.2270.4669.8869.3568.6668.2063.4659.0651.40 71.0769.6969.2069.0559.21 70.22              F1+F2+F3





   F                77.7075.7375.8074.9874.4875.5273.9574.1671.8170.9569.6570.0365.4260.6365.24 77.7176.5476.2775.7061.20 75.72 76.35 77.0574.8875.5975.2074.6974.8070.7461.5261.33 77.0276.1775.4374.8161.79 76.32 80.0579.1278.6979.7879.9979.2274.8762.7467.12 79.3179.9780.1779.7163.23 80.48




   P       BLANC        78.0778.9478.8174.7379.1577.2877.9276.8675.0868.3772.8371.6666.1458.7766.86 78.6079.7780.1980.0859.27 77.29 77.92 77.8773.9578.9980.2276.9373.5867.4259.5160.00 78.2880.2880.4173.6259.76 78.38 78.9984.8083.1080.1781.4782.9072.3761.1969.93 84.8480.3181.5984.4361.55 83.31



   R                77.3573.2973.4775.2471.3074.0371.1272.0569.4375.1667.4068.6964.7767.2364.01 76.8974.0773.4272.6366.84 74.36 74.98 76.2875.9273.0571.8572.8676.1977.2968.4963.26 75.8973.2272.0976.1668.10 74.60 81.2375.4175.6079.4178.6876.4978.6069.8765.26 75.6679.6578.9276.4270.28 78.23  English.

    F3                48.3746.6045.8744.8146.4143.6845.4045.2344.7738.9541.5340.0536.6834.2230.74 49.6647.9646.8447.6034.23 45.52 46.73 47.9745.1245.2746.7945.6141.8337.6034.5331.39 48.6546.0547.3841.9334.44 46.19 68.4657.6956.4654.2254.4153.2344.4243.5336.58 58.4054.7255.0754.3543.83 56.40 for




   P                                                                                                                                                                                             track      CEAFe       43.1744.7243.4243.1146.1544.8945.6744.7445.3532.6441.6435.9543.5534.9622.30 44.5146.6144.6847.2134.65 46.21 47.97 41.6941.7141.3544.9246.5345.8335.4534.6736.51 42.2342.3645.5546.3434.04 45.42 62.7146.4045.4642.5543.7441.5233.6737.1025.98 47.2643.0944.4842.6537.44 45.27



   R                55.0048.6448.6046.6446.6842.5345.1345.7444.2048.2941.4245.2031.6833.5249.44 56.1649.3849.2247.9933.82 44.85 45.55 56.4849.1450.0048.8144.7338.4640.0434.3827.53 57.3650.4549.3638.2834.85 46.99 75.3876.2574.4974.7171.9474.1465.2852.6461.82 76.4374.9372.2974.8852.84 74.77                                                                                                                                                                                                                                   closed


   F                61.9459.6159.2058.2858.1356.7657.2457.2855.6853.1952.4451.9044.8343.5441.97 63.0560.9259.9059.7043.71 58.35 59.55 61.5958.5158.4858.5757.9055.2052.0243.9639.30 62.0059.2059.3855.3043.90 59.04 73.7668.3567.7667.6466.7466.3259.1350.3547.82 68.8767.8367.2666.9050.61 68.14 the  RESOLUTION

   P                61.9459.6159.2058.2858.1356.7657.2457.2855.6853.1952.4451.9044.8343.5441.97 63.0560.9259.9059.7043.71 58.35 59.55 61.5958.5158.4858.5757.9055.2052.0243.9639.30 62.0059.2059.3855.3043.90 59.04 73.7668.3567.7667.6466.7466.3259.1350.3547.82 68.8767.8367.2666.9050.61 68.14 for      CEAFm


   R                61.9459.6159.2058.2858.1356.7657.2457.2855.6853.1952.4451.9044.8343.5441.97 63.0560.9259.9059.7043.71 58.35 59.55 61.5958.5158.4858.5757.9055.2052.0243.9639.30 62.0059.2059.3855.3043.90 59.04 73.7668.3567.7667.6466.7466.3259.1350.3547.82 68.8767.8367.2666.9050.61 68.14  COREFERENCE
    F2                71.2470.3670.2669.3468.9669.3168.5168.2767.0567.3465.9565.9961.3758.3555.98 71.7571.3970.8470.1658.48 69.71 70.58 70.8569.7569.7069.2369.2567.9567.2558.5659.51 70.9770.2769.8368.1058.82 70.22 77.4673.7573.3974.1972.8172.6770.6360.7457.18 74.2074.3873.2173.2160.80 73.89                                                                                                                                                                                                                                                                                                                                                        evaluations



   P                77.6974.6975.2471.8171.5269.5470.7170.4568.9073.2968.4872.7456.6754.9080.06 78.0175.3475.7172.7755.62 71.01 71.47 78.9573.8776.5973.5669.9964.9866.9755.6053.93 79.0977.0174.0264.9756.59 73.05 79.6985.9284.6885.6183.1687.0380.5758.6081.21 85.8785.5183.2887.5658.63 85.72         BCUBED


   R                65.7966.5065.9067.0466.5769.0966.4566.2365.3162.2863.5960.3866.9362.2643.03 66.4267.8366.5667.7361.66 68.45 69.70 64.2566.0763.9565.3868.5371.2167.5361.8566.37 64.3664.6266.0971.5561.22 67.60 75.3664.6064.7565.4664.7562.3762.8863.0444.13 65.3265.8165.3162.9063.14 64.94


    F1                70.5166.9767.5866.3863.7165.1063.8262.5562.9266.1860.8959.8448.2753.5250.84 72.1869.1169.0266.6554.02 65.85 67.83 70.6767.8067.2964.9763.6462.2664.5554.8238.21 71.4068.3166.3362.5955.09 67.27 85.7479.9479.8079.6578.7778.7075.3472.9260.45 80.6279.9779.3379.6073.01 80.38                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           supplementary



   P    MUC        75.9168.8370.1068.0663.9663.9863.5763.0262.3975.1860.7865.2042.7252.7972.57 77.2570.3771.1766.9953.59 65.22 66.73 77.5371.3671.6366.7862.7958.8067.1354.6733.66 78.2672.2968.0758.8255.51 68.00 88.4889.4089.1691.2188.1290.7189.9478.1181.51 89.6991.2488.4091.1578.10 90.04 and



   R
                65.8365.2165.2364.7763.4766.2664.0862.0863.4659.1161.0055.2955.4854.2839.12 67.7367.9067.0066.3054.45 66.50 68.96 64.9264.5863.4463.2664.5066.1662.1554.9644.17 65.6564.7564.6766.8854.68 66.55 83.1672.2972.2270.6971.2269.5164.8168.3848.04 73.2271.1871.9670.6668.54 72.59   primary
   F                77.7375.1575.3874.3273.8273.9273.7172.5373.6871.9670.8868.7962.7867.1159.93 79.4777.1076.6876.1367.79 74.39 76.18 78.0375.7275.3075.2473.2071.5970.7468.6151.08 78.8376.1376.0871.8168.65 75.86 100.0089.3988.91100.0087.7486.1680.3087.9872.75 89.78100.0088.1686.9788.10 88.61                                                                                              the       DETECTION P                                                           in                83.4576.1077.0976.2572.6072.3972.2471.8872.0884.5569.9673.9954.5568.4986.72 84.8177.3377.8675.1569.62 72.64 73.83 85.2979.8378.9775.7070.8467.3075.8270.9144.47 86.1179.2576.5867.1471.72 75.47 100.00100.00100.00100.00100.00100.00100.00100.0099.99 100.00100.00100.00100.00100.00 100.00




   R      MENTION          72.7574.2373.7572.4875.0875.5175.2373.1975.3562.6471.8264.2873.9365.7845.78 74.7676.8775.5377.1366.05 76.23 78.68 71.9172.0171.9574.7875.7376.4666.3166.4560.00 72.6973.2475.5977.1765.82 76.26 100.0080.8280.03100.0078.1775.6967.0978.5557.18 81.46100.0078.8376.9478.73 79.55  systems
     Qlty. GM                                          ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ of
      GB                         ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
TestMention NB ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      Syntax GA ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    Performance

   G                      ■ ■                ■                ■                                                                                              20:TrainSyntax A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
                                                                                                                                                            Table
           Participant                             fernandesmartschatbj¨orkelundchangchenstamborgchunyangyuanshouxuuryupinayangxinxinzhekovali  fernandesmartschatbj¨orkelundchenzhekova   bj¨orkelund   bj¨orkelund  fernandeschangbj¨orkelundchenyuanstamborgxuzhekovali  fernandesbj¨orkelundchenstamborgzhekova   bj¨orkelund changchenyuanfernandesstamborgbj¨orkelundxuzhekovali chenfernandesstamborgbj¨orkelundzhekova   bj¨orkelund


   ID                    e0.00 e0.01 e0.02 e0.03 e0.04 e0.05 e0.06 e0.07 e0.08 e0.09 e0.10 e0.11 e0.12 e0.13 e0.14   e1.00 e1.01 e1.02 e1.03 e1.04   e2.00   e3.00   e4.00 e4.01 e4.02 e4.03 e4.04 e4.05 e4.06 e4.07 e4.08   e5.00 e5.01 e5.02 e5.03 e5.04   e6.00   e7.00 e7.01 e7.02 e7.03 e7.04 e7.05 e7.06 e7.07 e7.08   e8.00 e8.01 e8.02 e8.03 e8.04   e9.00




                                       31
<a id="page-32"></a>

### PDF 第 32 页

3   Ofﬁcial          62.2460.6959.9759.2258.4956.8553.8753.1551.8351.7646.2745.7144.53 69.3866.4665.0863.7762.2860.1356.3456.0754.2154.0152.6647.5746.8546.12 57.64 61.61 68.5564.4264.0862.7661.4854.3044.9343.28 69.3865.0764.3962.2854.5246.0841.40 62.32 74.68 77.7776.0569.9269.7966.9766.3659.9351.44 78.7976.7372.4567.1266.4461.6152.70              F1+F2+F3





   F                77.3475.7676.0773.0077.1175.0073.1169.1866.4571.4468.9263.1662.79 80.8078.4175.8177.7779.6078.0574.4270.9566.8072.9067.9569.9663.7963.73 75.48 77.53 80.3277.0574.8077.4079.2272.6962.3461.47 80.8077.3778.3679.6074.0263.8159.83 78.02 82.09 84.0981.8181.5676.4879.8981.7173.4765.58 84.4582.0681.6680.9181.5573.7266.45




   P       BLANC        84.5378.2281.6174.5779.5977.8480.7682.8165.8373.9176.6371.2461.64 87.0779.8475.1782.3681.2880.0980.8784.8665.0374.6673.9177.3971.5762.15 77.96 78.94 86.7578.4877.0685.4980.9671.1261.4758.70 87.0778.7985.5281.2872.5362.3157.08 81.33 89.40 91.3888.1491.5679.7183.7083.3977.6763.62 91.5588.3091.1185.6583.2176.9264.06



   R                72.7973.7272.2571.6475.0672.7468.7263.9667.1269.4864.9959.8264.29 76.4877.1276.5074.3878.1176.2970.3765.3969.1671.4064.5166.0160.4065.98 73.42 76.26 75.9575.7772.9072.4477.6974.5363.4167.82 76.4876.1073.7078.1175.7565.9168.12 75.38 77.30 79.2177.4875.7773.9876.9680.2170.5368.60 79.6377.7776.0677.4380.0871.3170.49  Chinese.

    F3                50.9748.8348.1947.4444.4442.4140.3239.9240.0038.8934.6235.9732.58 58.0053.8652.2651.2847.6645.0642.4542.0641.8840.8739.8135.4236.8033.50 47.15 50.26 57.1751.8151.3750.2047.2940.9133.0731.86 58.0052.3052.0147.6641.2333.4930.15 50.99 63.77 68.3865.5856.6156.1153.2151.3646.8337.37 69.7466.4059.9353.4851.2149.0337.96 for




   P      CEAFe       48.7350.7048.2944.8137.8838.0434.5232.4640.1738.5325.2429.9925.24 57.6458.3850.7252.9241.5041.1537.4934.4043.4742.1839.5225.7731.1026.13 53.10 59.31 56.4157.4749.0748.5142.0247.7125.7143.51 57.6458.1351.8241.5048.0526.1345.72 58.43 52.45 57.8654.6544.2443.7742.0139.6836.0627.08 59.4555.6047.7342.2239.4938.8527.69  track



   R                53.4347.1048.0950.4053.7547.9048.4751.8139.8439.2655.1144.9245.92 58.3749.9953.9049.7455.9749.8048.9454.1140.3939.6540.1156.5845.0646.68 42.39 43.61 57.9547.1753.9052.0354.0935.8146.3525.14 58.3747.5452.2155.9736.1146.6222.49 45.23 81.32 83.5681.9878.5978.1372.5672.7866.7660.27 84.3382.4080.5372.9372.8466.4460.37                                                                                                                                                                                                                                                                    closed


   F                62.1859.5959.0157.4657.7355.5752.4051.3049.8849.9245.7044.8941.86 68.6764.7762.5362.1761.1958.9854.5953.9251.9251.9549.6146.7945.9443.34 57.39 60.80 67.8762.9661.5761.3260.4052.8242.0941.06 68.6763.5262.9161.1953.2643.2939.32 61.52 72.24 75.8373.6768.3066.2265.6065.1057.5046.88 76.7674.2370.3065.8965.0658.7447.84 the  RESOLUTION

   P                62.1859.5959.0157.4657.7355.5752.4051.3049.8849.9245.7044.8941.86 68.6764.7762.5362.1761.1958.9854.5953.9251.9251.9549.6146.7945.9443.34 57.39 60.80 67.8762.9661.5761.3260.4052.8242.0941.06 68.6763.5262.9161.1953.2643.2939.32 61.52 72.24 75.8373.6768.3066.2265.6065.1057.5046.88 76.7674.2370.3065.8965.0658.7447.84 for      CEAFm


   R                62.1859.5959.0157.4657.7355.5752.4051.3049.8849.9245.7044.8941.86 68.6764.7762.5362.1761.1958.9854.5953.9251.9251.9549.6146.7945.9443.34 57.39 60.80 67.8762.9661.5761.3260.4052.8242.0941.06 68.6763.5262.9161.1953.2643.2939.32 61.52 72.24 75.8373.6768.3066.2265.6065.1057.5046.88 76.7674.2370.3065.8965.0658.7447.84  COREFERENCE
    F2                73.5572.9073.1072.1670.7070.3868.3167.1365.6668.3159.5463.2360.45 77.6876.1375.4075.0273.1072.5869.7669.0066.3369.2469.0759.9563.6561.41 71.45 73.21 77.0475.0274.8574.5272.6568.7360.3259.27 77.6875.4675.4973.1068.9961.3556.25 74.00 78.93 81.1579.7976.3075.9573.8474.3066.8461.94 81.9180.3077.9774.0074.4367.5963.04                                                                                                                                                                                                                                                                                                                                                                                                           evaluations



   P                77.8172.6775.0776.3880.5777.5580.7885.2465.5070.6086.0678.3177.65 80.2074.3777.6575.8481.4378.5480.1186.6964.4969.7872.9186.0677.6077.28 67.35 67.83 79.9172.3078.7079.1479.8863.0977.3848.52 80.2072.6878.3881.4363.4177.2743.67 69.93 91.38 91.4391.2193.5191.2388.1289.4382.9980.81 91.6791.3993.4888.5689.7680.2580.53         BCUBED


   R                69.7373.1271.2368.3962.9964.4359.1755.3765.8166.1645.5153.0249.49 75.3277.9773.2874.2166.3167.4761.7857.3168.2768.7065.6146.0053.9650.95 76.08 79.51 74.3777.9671.3670.4266.6275.4749.4376.15 75.3278.4672.7966.3175.6450.8779.00 78.57 69.47 72.9570.9164.4465.0563.5463.5555.9550.22 74.0271.6266.8863.5663.5758.3751.79


    F1                62.2160.3358.6158.0760.3457.7752.9752.4149.8348.0944.6537.9340.56 72.4669.3867.5865.0166.0762.7656.8157.1554.4151.9249.1147.3340.1143.44 54.32 61.36 71.4366.4466.0363.5664.4953.2741.3938.70 72.4667.4465.6666.0753.3343.3937.81 61.98 81.34 83.7782.7976.8577.3273.8673.4366.1355.00 84.7183.4879.4673.8773.6968.2057.10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    supplementary


   P
    MUC        64.6958.4258.4961.4770.5864.1363.1367.8049.6448.5571.4449.2257.97 72.7765.5069.2863.3674.4967.9564.5372.1952.8150.4449.4973.7750.2960.29 48.92 54.07 72.1261.5968.7365.5471.4647.2058.3030.62 72.7762.4965.8774.4947.2760.1928.50 55.71 92.28 92.4192.7493.2394.0788.2390.8182.2879.57 92.8593.0693.7688.2491.4581.0480.89 and



   R
                59.9262.3658.7255.0252.6952.5645.6242.7150.0247.6432.4830.8531.19 72.1473.7365.9566.7659.3558.3250.7447.3056.1153.4948.7434.8533.3633.95 61.06 70.92 70.7672.1263.5461.6958.7661.1332.0952.56 72.1473.2565.4659.3561.1733.9356.15 69.85 72.72 76.6074.7765.3665.6363.5261.6455.2842.02 77.8875.6968.9463.5261.7058.8844.12   primary   F                71.6468.1566.3765.2066.1364.0159.0358.6061.6155.8951.5347.5847.32 81.1976.4374.0372.5771.6868.6163.1462.8066.4160.3257.5755.6651.0350.27 63.46 70.36 80.4574.0272.9471.0270.9162.2449.3051.90 81.1974.7273.2771.6862.3250.3052.14 70.83 89.55 91.7389.8583.4783.3881.63100.0077.7364.43 92.4290.3886.3281.71100.0081.4365.61
                                                                                                            the       DETECTION P                72.1664.0963.5466.1078.2871.9370.0974.0262.1256.0987.0159.9772.52 79.2570.3373.0768.0381.7375.2171.3877.1765.3358.2156.5289.7462.1674.84 55.07 59.87 78.9766.8672.6870.1879.1855.2873.9841.88 79.2567.4670.2881.7355.3274.7940.23 61.19 100.00 100.0099.79100.00100.00100.00100.0099.95100.00 100.0099.80100.00100.00100.00100.00100.00 in




   R      MENTION          71.1272.7569.4564.3357.2457.6550.9848.4961.1155.6836.6039.4335.12 83.2283.6975.0277.7763.8363.0756.6152.9467.5262.5958.6640.3443.2837.84 74.86 85.29 81.9982.8973.2171.8864.2171.2136.9768.22 83.2283.7576.5363.8371.3637.8974.10 84.09 81.07 84.7281.7271.6371.5168.97100.0063.5947.53 85.9282.5875.9369.08100.0068.6848.82  systems
     Qlty. GM                                                  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ of
      GB                                ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
TestMention NB ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      Syntax GA ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    Performance

   G                             ■ ■                 ■ ■                                                                                                            21:TrainSyntax A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
                                                                                                                                                                                   Table
           Participant             chenyuanbj¨orkelundxufernandesstamborguryupinamartschatchunyangxinxinlichangzhekova chenyuanxubj¨orkelundfernandesstamborguryupinamartschatchunyangxinxinyanglichangzhekova   bj¨orkelund   bj¨orkelund chenyuanxubj¨orkelundfernandesstamborgzhekovali chenyuanbj¨orkelundfernandesstamborgzhekovali   bj¨orkelund   bj¨orkelund chenyuanbj¨orkelundxustamborgfernandeslizhekova chenyuanbj¨orkelundstamborgfernandeslizhekova


   ID                    c0.00 c0.01 c0.02 c0.03 c0.04 c0.05 c0.06 c0.07 c0.08 c0.09 c0.10 c0.11 c0.12   c1.00 c1.01 c1.02 c1.03 c1.04 c1.05 c1.06 c1.07 c1.08 c1.09 c1.10 c1.11 c1.12 c1.13   c2.00   c3.00   c4.00 c4.01 c4.02 c4.03 c4.04 c4.05 c4.06 c4.07   c5.00 c5.01 c5.02 c5.03 c5.04 c5.05 c5.06   c6.00   c7.00   c8.00 c8.01 c8.02 c8.03 c8.04 c8.05 c8.06 c8.07   c9.00 c9.01 c9.02 c9.03 c9.04 c9.05 c9.06




                                      32
<a id="page-33"></a>

### PDF 第 33 页

3   Ofﬁcial          54.2253.5550.4149.4347.1340.5733.53 55.3853.8447.1336.82 53.22 55.82 53.9053.5049.5947.2740.2431.46 55.3853.8449.4947.1336.86 55.83 63.4959.1455.7253.3552.2640.62 63.2358.9056.3553.8852.32 62.82              F1+F2+F3





   F                66.9769.6367.6966.8763.6960.6554.12 67.9469.6563.7855.63 69.06 71.85 66.6469.6166.4263.8759.8651.46 67.9469.6565.0663.7855.66 71.86 71.4974.6166.1269.8766.9057.96 71.9374.8463.2869.2966.85 76.41




   P       BLANC        71.9174.6170.5666.9461.8479.1973.93 72.9773.8361.8661.78 70.71 72.67 70.0973.4366.0861.9476.3551.10 72.9773.8464.8961.8661.95 72.70 74.8780.8582.0073.4666.6568.52 75.5180.9272.0772.3266.52 80.61



   R                63.9866.4565.5866.8066.4557.1052.91 64.8466.8166.7054.13 67.69 71.09 64.2866.9466.7966.7856.6154.04 64.8466.8165.2366.7054.15 71.10 69.0370.6961.3667.3767.1555.64 69.3570.9759.9467.1067.19 73.35  Arabic.
    F3                                                                 for                49.0844.3042.2842.4940.8434.5330.41 49.5644.4441.0233.65 45.06 47.11 48.5344.0042.4641.2234.8228.84 49.5644.4442.1641.0233.63 47.10 56.3249.3246.6645.3746.4334.81 55.9049.0248.9145.6646.46 53.71
   P                                                                                                                                    track      CEAFe       46.0940.8042.1340.3639.8424.8120.95 46.8841.5340.2724.22 44.33 47.98 47.3941.2440.5040.9025.3642.87 46.8841.5140.4740.2624.20 47.95 46.0037.9934.5234.5236.2424.36 45.5837.7137.5434.8536.27 42.75



   R                52.4948.4542.4344.8641.8956.7955.45 52.5747.7841.8055.10 45.82 46.26 49.7347.1644.6041.5555.5321.73 52.5747.8044.0141.8155.10 46.27 72.6170.2871.9666.1664.6060.95 72.2470.0170.1766.2164.59 72.24                                                                                                                                                                           closed



   F                55.5953.4250.8250.1647.4942.7437.03 56.4953.5247.7339.52 53.77 56.21 54.8853.1849.9247.8442.5733.68 56.4953.5249.5547.7339.52 56.21 62.5659.5055.4254.0053.1642.25 62.6259.4155.1154.1253.19 63.12 the  RESOLUTION   P                                                                 for                55.5953.4250.8250.1647.4942.7437.03 56.4953.5247.7339.52 53.77 56.21 54.8853.1849.9247.8442.5733.68 56.4953.5249.5547.7339.52 56.21 62.5659.5055.4254.0053.1642.25 62.6259.4155.1154.1253.19 63.12      CEAFm


   R                55.5953.4250.8250.1647.4942.7437.03 56.4953.5247.7339.52 53.77 56.21 54.8853.1849.9247.8442.5733.68 56.4953.5249.5547.7339.52 56.21 62.5659.5055.4254.0053.1642.25 62.6259.4155.1154.1253.19 63.12  COREFERENCE
    F2                67.1168.5467.4664.6161.5357.3352.14 67.6668.6861.4853.65 67.73 69.71 66.9168.6164.2261.6557.7454.25 67.6668.6864.1361.4853.67 69.72 68.6867.2964.9262.2660.0853.74 68.6267.2863.9562.6760.13 69.81   evaluations




   P                72.1975.3269.2367.9562.5190.7293.34 72.3774.4562.0684.86 69.74 69.88 69.4474.2767.2461.7789.2541.21 72.3774.5066.7562.0784.95 69.92 79.8185.3589.6981.3075.2588.07 80.0285.4082.3580.8375.19 83.75         BCUBED


   R                62.7062.8965.7761.5760.5941.9136.17 63.5363.7360.9139.22 65.83 69.53 64.5663.7561.4561.5242.6779.37 63.5363.7161.7060.9039.23 69.52 60.2755.5550.8750.4550.0038.67 60.0755.5152.2751.1750.10 59.85


    F1                46.4647.8241.4941.1839.0229.8518.05 48.9348.3938.8923.15 46.87 50.65 46.2647.9042.1038.9528.1611.30 48.9348.4142.1838.8923.28 50.67 65.4860.8155.5852.4350.2833.31 65.1760.4156.2053.3050.36 64.94                                                                                                                                                                                                                                                                                                                                                                                  supplementary



   P    MUC                49.6952.5141.6643.4939.9662.1355.60 51.7852.1539.5745.92 47.66 49.76 47.3951.4744.1739.2456.477.78 51.7852.2043.9839.5946.18 49.80 76.4878.6280.3669.7863.2364.62 76.2778.3173.2770.4263.28 78.84 and



   R                43.6343.9041.3339.1138.1319.6410.77 46.3845.1438.2215.47 46.11 51.57 45.1844.7840.2238.6618.7520.62 46.3845.1440.5338.2215.56 51.57 57.2549.5742.4841.9941.7222.43 56.8949.1745.5842.8841.81 55.21                                                                                                                                                                                                       primary
   F                64.7960.5555.3959.4759.8041.0229.65 66.8261.3059.7041.78 62.20 64.67 65.0860.6160.8159.7640.2929.78 66.8261.3060.7659.7241.87 64.67 100.0076.4373.3871.9073.6552.58 100.0075.8179.2972.3873.63 81.30                                                                      the       DETECTION P                                            in                67.0064.8654.3563.2863.9580.3480.43 68.7164.6063.4182.21 62.52 62.44 64.8263.7464.6262.5575.5320.71 68.7164.6364.1863.4582.39 62.47 100.00100.00100.00100.00100.00100.00 100.00100.00100.00100.00100.00 100.00




   R      MENTION          62.7256.7856.4756.1056.1627.5418.17 65.0358.3356.4128.00 61.88 67.07 65.3457.7757.4357.2127.4852.95 65.0358.2957.6856.4128.06 67.04 100.0061.8557.9556.1358.2935.67 100.0061.0565.6856.7258.26 68.50  systems
                                 ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ of
     Qlty. GM
      GB                ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
TestMention NB ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      Syntax GA ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    Performance
   G             ■ ■             ■             ■ 22:
TrainSyntax A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
                                                                                                                     Table
           Participant                             fernandesbj¨orkelunduryupinastamborgchenzhekovali  fernandesbj¨orkelundchenzhekova   bj¨orkelund   bj¨orkelund  fernandesbj¨orkelundstamborgchenzhekovali  fernandesbj¨orkelundstamborgchenzhekova   bj¨orkelund  fernandesbj¨orkelundzhekovastamborgchenli  fernandesbj¨orkelundzhekovastamborgchen   bj¨orkelund


   ID                    a0.00 a0.01 a0.02 a0.03 a0.04 a0.05 a0.06   a1.00 a1.01 a1.12 a1.13   a2.00   a3.00   a4.00 a4.01 a4.02 a4.03 a4.04 a4.05   a5.00 a5.01 a5.02 a5.03 a5.04   a6.00   a7.00 a7.01 a7.02 a7.03 a7.04 a7.05   a8.00 a8.01 a8.02 a8.03 a8.04   a9.00




                              33
<a id="page-34"></a>

### PDF 第 34 页

7.3  Arabic Closed                                 7.4  All Languages Open
  Table 22 shows the performance for the Arabic     Tables 24, 25 and 26, give the performance for
language in greater detail.                             the systems that participated in the open track. Not
                                        many systems participated in this track, so there is
7.3.1  Ofﬁcial Setting                              not a lot to observe. One thing to note is that chen
  Unlike English and Chinese, none of the system   modiﬁed precise constructs sieve to add named en-
was particularly tuned for Arabic. This gives us an    tity information in the open track sieve which gave
unique opportunity to test the performance variation   them a point improvement in performance.  With
of a mostly statistical, roughly language indepen-   gold mentions and gold syntax during testing the
dent mechanism. Although, there could possibly be   chen system performance almost approaches an F-
a signiﬁcant bias that Arabic language brings to the   score of 80 (79.79)
mix. The overall performance for Arabic seems to
be about ten points below both English and Chinese.   7.5  Headword-based and Genre speciﬁc scores
On the mention detection front, most of the systems     Since last year’s task showed that there was only
have a balanced precision and recall, and the drop   some very local difference in ranking between sys-
in performance seems quite steady. bj¨orkelund has   tems scored using the strict boundaries versus the
a slight edge on fernandes on the MUC, BCUBED   ones using headword based scoring, we did not com-
and BLANC metrics, but fernandes has a much larger   pute the headword based evaluation.
lead on both the CEAF metrics, putting it on the top    Owing to space constraints, we cannot present a
in the ofﬁcial score. We haven’t reported the de-   detailed analysis of the variation across genre. How-
velopment set numbers here, but another thing to   ever, since genre variation is important to note, we
note especially for Arabic is that performance on   present the performance of the highest performing
Arabic test set is signiﬁcantly better than on the de-   system across all the three languages and genres in
velopment set as pointed out by bj¨orkelund.  This   Table 23. For each language there are three logical
is probably because of the smaller size of the train-   performance blocks:  i) The ofﬁcial, predicted ver-
ing set and therefore a higher relative increment over   sion, with no provided boundaries is the ﬁrst block;
training set. The size of the training set (which is    ii) The supplementary version with gold mention
roughly about a third of either Engish or Chinese)   boundaries is the second block; and iii) The third
also could itself be a factor that explains the lower   block shows the performance for the supplementary
performance, and that Arabic performance might   version given gold mentions.
gain from more data. chen did not use development     Looking at the Engish performance on the ofﬁcial,
data for the ﬁnal models. Using that could have in-   closed track, there seems to be a cluster of genre –
creased their score.                               BC, BN, NW and WB – where the performance is very
                                                      close to a score of 60. Whereas, genres TC, MZ and
7.3.2  Gold Mention Boundaries                                             PT are increasingly better. Surprisingly, a simplistic
  The system performance given gold boundaries                                                  look at the individual metrics does indicate a similar
followed more of the trend in English than Chinese.                                                          trend, except for the CEAFe score for the TC and WB
There was not much improvement over the primary                                                  being somewhat reversed.  It so happens that these
NB evaluation. Interestingly, chen uses gold bound-                                                      the two genres — MZ and PT – are professional hu-
aries for Chinese so well, but does not get any per-                                       man translations from a foreign language. As seen
formance improvement. This might indicate that the                                                                earlier, there is not a huge shift in performance when
technique that helped that system in Chinese does                                                      the systems are provided with gold mention bound-
not generalize well across languages.                                                           aries. However, when provided with gold mentions
7.3.3  Gold Mentions                                there is a big improvement in performance across
                                                      the board. Especially so with MZ genre for which  Performance given gold mentions seems to be
                                                      the improvement is more than double (9.5 points)about  ten  points  higher  than  in  the NB  case.
                                                   over the improvement in PT genre (3.5 points) withbj¨orkelund does well on BLANC metric than fernan-
                                                      the most notable improvement (of 5 points) in thedes even after getting a big hit in recall for mention
                                           CEAFe metric, which also is another indication thatdetection. In absence of chang, it seems like fernan-
                                                             this metric does a good job of rewarding correctdes is the only one that explicitly adds a constraint
                                                  anaphoric mentions.for the GM case and gets a perfect mention detec-
                                                Looking at the Chinese performance, we see thattion score. All other systems loose signiﬁcantly on
                                                      the NW genre does particularly worse than all oth-recall.
                                                         ers on the ofﬁcial, closed track. The BC genre does
7.3.4  Gold Test Parses                        somewhat worse than WB, MZ, and TC all of which
   Finally, providing gold parses during testing does   seem to be around the same ballpark, with BN lead-
not have much of an impact on the scores.            ing the pack. Again, provided gold mention bound-

                                        34
<a id="page-35"></a>

### PDF 第 35 页

Train                                                                       Test
                          Genre             Syntax                                                      Syntax  Mention                                                                                    Qlty.   MD   MUC  BCUBEDCOREFERENCECEAFmRESOLUTIONCEAFe  BLANC       Ofﬁcial
                                        A  G  A  G  NB  GB  GM      F      F1      F2         F       F3        F    F1+F2+F33
                                                                  ENGLISH
                     Pivot Text [PT]         ■   ■    ■               89.13   82.49     72.66     68.92    54.47    79.20        69.87
                 Magazine [MZ]         ■   ■    ■               77.70   69.57     77.29     68.88    57.07    81.84        67.98
                  Telephone Conversation [TC]   ■   ■    ■               79.95   76.75     72.31     62.06    43.22    79.24        64.09
                 Weblogs and Newsgroups [WB]  ■   ■    ■               78.21   71.66     68.61     59.42    45.24    76.42        61.84
                   Broadcast News [BN]      ■   ■    ■               74.60   65.15     70.60     60.90    49.52    74.45        61.76
                   Brodcast Conversation [BC]   ■   ■    ■               75.67   67.54     69.14     57.70    44.99    76.74        60.56
                 Newswire [NW]         ■   ■    ■               71.24   62.67     71.01     60.61    47.73    75.40        60.47
                     Pivot Text [PT]         ■   ■      ■          89.50   82.74     72.65     68.98    54.28    79.42        69.89
                 Magazine [MZ]         ■   ■      ■          77.27   68.68     76.53     67.51    55.63    79.72        66.95
                  Telephone Conversation [TC]   ■   ■      ■          81.95   78.18     72.53     63.33    44.32    77.99        65.01
                 Weblogs and Newsgroups [WB]  ■   ■      ■          79.08   72.62     68.94     60.09    45.74    76.46        62.43
                   Broadcast News [BN]      ■   ■      ■          75.10   65.56     69.98     60.47    49.14    74.10        61.56
                   Brodcast Conversation [BC]   ■   ■      ■          75.96   67.64     68.51     57.14    44.85    74.81        60.33
                 Newswire [NW]         ■   ■      ■          70.44   61.63     70.04     59.44    46.57    73.51        59.41
                 Magazine [MZ]         ■   ■         ■   100.00   82.87     83.10     78.02    66.93    87.00        77.63
                     Pivot Text [PT]         ■   ■         ■   100.00   86.20     74.30     71.67    59.43    80.12        73.31
                  Telephone Conversation [TC]   ■   ■         ■   100.00   84.74     75.18     66.29    49.68    77.37        69.87
                 Weblogs and Newsgroups [WB]  ■   ■         ■   100.00   82.38     71.43     66.08    53.28    77.96        69.03
                 Newswire [NW]         ■   ■         ■   100.00   74.00     74.41     67.39    53.28    81.03        67.23
                   Broadcast News [BN]      ■   ■         ■   100.00   74.51     73.31     65.71    52.96    79.15        66.93
                   Brodcast Conversation [BC]   ■   ■         ■   100.00   77.52     71.49     63.73    50.54    79.59        66.52

                                                                   CHINESE
                   Broadcast News [BN]      ■   ■    ■               78.02   71.71     78.80     68.93    55.87    83.85        68.79
                 Weblogs and Newsgroups [WB]  ■   ■    ■               79.29   71.30     71.05     60.68    46.81    80.94        63.05
                 Magazine [MZ]         ■   ■    ■               75.34   70.26     72.32     62.63    46.42    81.34        63.00
                  Telephone Conversation [TC]   ■   ■    ■               79.79   72.58     71.14     61.16    43.78    76.82        62.50
                   Brodcast Conversation [BC]   ■   ■    ■               73.80   64.22     67.68     55.38    42.89    72.98        58.26
                 Newswire [NW]         ■   ■    ■               52.38   49.74     67.97     54.82    43.79    75.63        53.83
                   Broadcast News [BN]      ■   ■      ■          78.02   71.71     78.80     68.93    55.87    83.85        68.79
                 Weblogs and Newsgroups [WB]  ■   ■      ■          79.29   71.30     71.05     60.68    46.81    80.94        63.05
                 Magazine [MZ]         ■   ■      ■          75.34   70.26     72.32     62.63    46.42    81.34        63.00
                  Telephone Conversation [TC]   ■   ■      ■          79.79   72.58     71.14     61.16    43.78    76.82        62.50
                   Brodcast Conversation [BC]   ■   ■      ■          73.80   64.22     67.68     55.38    42.89    72.98        58.26
                 Newswire [NW]         ■   ■      ■          52.38   49.74     67.97     54.82    43.79    75.63        53.83
                   Broadcast News [BN]      ■   ■         ■   100.00   81.03     81.34     75.07    62.99    86.18        75.12
                  Telephone Conversation [TC]   ■   ■         ■   100.00   87.31     77.80     71.01    59.44    78.78        74.85
                 Weblogs and Newsgroups [WB]  ■   ■         ■   100.00   80.36     72.46     64.49    51.93    81.10        68.25
                 Magazine [MZ]         ■   ■         ■   100.00   75.18     73.12     65.90    48.81    84.17        65.70
                   Brodcast Conversation [BC]   ■   ■         ■   100.00   76.42     70.01     61.75    49.81    74.14        65.41
                 Newswire [NW]         ■   ■         ■   100.00   51.42     67.83     55.29    43.81    76.53        54.35

                                                                  ARABIC
                 Newswire [NW]         ■   ■    ■               64.79   46.46     67.11     55.59    49.08    66.97        54.22
                 Newswire [NW]         ■   ■      ■          65.08   46.26     66.91     54.88    48.53    66.64        53.90
                 Newswire [NW]         ■   ■         ■   100.00   65.48     68.68     62.56    56.32    71.49        63.49
        Table 23: Per genre performance for fernandes on the closed, ofﬁcial and supplementary evaluations.


aries there is very little or no change in performance.  8  Comparison with CoNLL-2011
And, when given the gold mentions the performance     Table 27 shows the performance of the systems
again shoots up by a signiﬁcant margin. Here again,   on CoNLL-2011 test subset which included only the
we see that the delta improvement in one particu-   English portion of OntoNotes v4.0. For the English
lar genre TC – is much higher (12 points) than in   subset, the size of training data in CoNLL-2011
BN (6 points), and once again the most improvement   was roughly 76% of CoNLL-2012 training data (1M
among all the metrics happens to be for the CEAFe.   vs 1.3M words respectively).  Although the mod-
Extremely surprising is the fact that the NW genre    els used to generate this table were trained on the
shows the lowest improvement among all genre. In  CoNLL-2012 English data and therefore on about
fact, the performance drops for the BCUBED met-  200K more words, it is still a small fraction of the
ric. This might have something to do with the fact    total training data.  In the past, coreference scores
that Chinese NW genre gets the lowest ITA among   have shown to asymptote after a small fraction of
all other (see Table 1), but then the better scoring TC   the total training data. Therefore, the 5% absolute
genre which has the second lowest ITA does con-   gap between the best performing systems of last year
siderably better (leading by roughly 10 points in the   can be attributed to algorithmic improvement, and
ofﬁcial setting, and 20 points in the gold mentions   possibly better rules. Given that a 200K data addi-
settings with respect to the TC genre). It could also   tion to a 1M word corpus is unlikely to help iden-
be possible that this has something to do with the    tify novel rules, and given that bj¨orkelund reported
fact (and pointed out earlier when discussing chen’s   adding (about 160K) development data (to the train-
results) that there is some overlapping mentions that   ing portion) to train the ﬁnal model had very lit-
were mistakenly included in the release.                  tle improvement in performance over using just the
  As for Arabic, since there was only one NW genre,   training data by itself, the possibility that the gain
there is nothing more to be analyzed. We plan to    is from algorithmic improvements seems even more
report more detailed tables and analysis on the task   plausible.
webpage.                                                               It is interesting to note that although the winning

                                        35
<a id="page-36"></a>

### PDF 第 36 页

3             3                         3             3   Ofﬁcial          59.23 60.74                                     Ofﬁcial          63.5361.0244.35 70.7866.3845.57 70.00 70.78 78.98 79.79                                     Ofﬁcial          44.37 45.31                                     Ofﬁcial          57.79 62.0960.4360.0959.3658.7158.1458.3657.8057.2556.3455.4054.1947.7747.8745.18              F1+F2+F3                                                                             F1+F2+F3                                                                                                                                                        F1+F2+F3                                                                             F1+F2+F3





   F           F                     F           F                73.94 74.78                                    78.9476.0260.59 82.2478.4960.90 81.88 82.24 85.41 85.67                                    58.46 59.20                                    73.02 76.8075.3274.9474.7274.0674.1673.7274.9570.9271.3069.8968.2965.2360.2164.12




   P           P                     P           P       BLANC        77.90 78.58                                     BLANC        84.2977.9967.08 86.6379.4567.37 86.39 86.63 90.73 91.00                                     BLANC        60.75 61.39                                     BLANC        76.21 78.5378.0477.6474.1378.6975.6776.9178.6973.3970.0572.2571.2764.7958.3767.02 set.                            English.
   R              R                                                       Chinese.     R             Arabic.     R                                    test                71.12 72.03                                    75.1574.3257.98 78.8777.6058.24 78.44 78.87 81.46 81.71                                    57.05 57.78                                    70.63 75.3173.1772.8175.3570.8972.8671.3172.2069.0072.8068.0766.2065.7165.4962.31           for                             for              for

    F3                   F3                                     F3                   F3
                45.36 46.54 track                          51.2749.0234.93 58.5553.6635.75 57.77 58.55 69.44 70.55 track                          42.97 43.03 track                          45.48 47.6346.2345.2344.2946.0542.9845.0844.9244.4238.7541.2639.8736.3834.5431.12

   P           P                     P           P      CEAFe       45.64 47.53                               CEAFe       49.1051.5228.12 58.4658.7028.83 57.32 58.46 59.17 60.42                               CEAFe       42.82 43.59                               CEAFe       47.75 42.5044.5542.6742.8246.1344.1345.7044.3845.3532.1741.8335.9843.5035.0922.69                  open                                                open                        open                                                                                                                        CoNLL-2012
   R           R                     R           R                45.09 45.60                                    53.6446.7546.10 58.6449.4147.04 58.22 58.64 84.03 84.77                                    43.12 42.48                                    43.41 54.1848.0548.1245.8645.9841.8944.4845.4843.5248.7140.7144.6931.2734.0149.52           the                             the              the                                    the


   F              F                                     F                                                 F                57.23 58.58 for                                                                63.4860.0542.71 70.1364.8443.66 69.40 70.13 77.50 78.26 for                                                                                                                                                          47.31 47.73 for                                                                                                                                                                                                         56.37 60.6158.8457.9957.4957.4755.3756.6256.9554.8852.8152.2551.0744.6843.4041.88 of  RESOLUTION                                                                                                                                     RESOLUTION                                                                                                                                                                                                                                                                    RESOLUTION                                                                                                                                     RESOLUTION

   P           P                     P           P                57.23 58.58                                    63.4860.0542.71 70.1364.8443.66 69.40 70.13 77.50 78.26                                    47.31 47.73                                    56.37 60.6158.8457.9957.4957.4755.3756.6256.9554.8852.8152.2551.0744.6843.4041.88      CEAFm                                                       CEAFm                                                                                                           CEAFm                                                       CEAFm   R           R                     R           R                                                                                  (English)                57.23 58.58                                    63.4860.0542.71 70.1364.8443.66 69.40 70.13 77.50 78.26                                    47.31 47.73                                    56.37 60.6158.8457.9957.4957.4755.3756.6256.9554.8852.8152.2551.0744.6843.4041.88                                       evaluations                                                                                                           evaluations                                                    evaluations  COREFERENCE                                                                                                                                                   COREFERENCE                                                                                                                                                                                                                                                                                              COREFERENCE                                                                                                                                                   COREFERENCE
    F2                   F2                                     F2                   F2                68.52 69.55                                    74.6173.0960.30 78.9376.1060.37 78.34 78.93 82.50 83.11                                    61.83 62.36                                    68.31 70.9570.6070.1669.8069.3069.1468.8468.9467.2267.3566.6266.0961.8159.0156.36                                                                                                                                                                                                                                       portion

   P           P                     P           P                70.69 71.14                                    78.3572.1677.45 80.8073.7477.38 80.49 80.80 91.59 91.94                                    62.81 62.55                                    68.23 78.0574.5075.3271.9771.3769.1970.3171.3868.3475.9368.2673.0256.1656.5381.37         BCUBED                                                                                BCUBED                                                                                                                                                            BCUBED                                                                                BCUBED


   R           R                     R           R                66.47 68.03                                    71.2174.0449.37 77.1478.6149.50 76.30 77.14 75.04 75.83                                    60.89 62.16                                    68.40 65.0367.0965.6667.7667.3469.1067.4366.6666.1460.5065.0560.3768.7361.7143.11                                                         supplementary                                                                                                                                                            supplementary                                                                            supplementary
    F1                   F1                                     F1                   F1                63.82 66.14                                    64.7060.9637.83 74.8569.3740.60 73.88 74.85 84.99 85.72                                    28.31 30.54                                    59.57 67.6864.4764.8763.9960.7762.2961.1759.5360.1262.9358.3156.6045.1350.0748.06           and                             and              and                                                                                                                       CoNLL-2011


   P           P                     P           P    MUC        63.57 65.27                      MUC        67.0858.4851.20 74.9365.1153.89 74.28 74.93 93.19 93.59                      MUC        28.43 30.10                      MUC        57.53 73.4666.2367.7265.5460.7061.1660.5760.0959.2473.2857.7261.8839.3749.5170.53 the
   R           R                     R           R                  on                64.08 67.03   primary                          62.4863.6730.00 74.7774.2432.57 73.50 74.77 78.12 79.07   primary                          28.20 30.99   primary                          61.76 62.7462.8062.2562.5160.8463.4761.7858.9961.0255.1458.9252.1552.8850.6436.45
                                                            the           the                                           the   F  73.69 75.58                                                                72.4468.4949.82 81.9576.2453.42 81.24 81.95 91.77 92.31                                                                                                                                                          53.89 55.17                                        in     F  70.70 75.2872.7672.9071.6171.3071.2771.1769.6471.1568.7768.1965.9059.4064.4257.40       in     F                             in     F                                                                                                                                                                                                                                       systems                                                                                                                                        DETECTION P       DETECTION P                                                                                                                                                                                                                                                                                                                                                                                            DETECTION P                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             DETECTION P                                                                                                   the                72.23 73.44                                                                73.4563.9767.55 80.4469.8571.10 80.11 80.44 100.00 100.00                                                                                                                                                          52.75 52.98                                                                                                                                                                                                         66.81 81.2773.5774.9173.3669.5669.7869.3168.8769.1682.6166.8070.9750.9865.7384.81                                                                                                                                            systems                         systems                                                                                                    systems   R              R                                     R                                                 R      MENTION                                                                                                 all                                                                                                          MENTION                                                                                                                                                                                                                                                                                                       MENTION                75.22 77.85                                                                71.4573.7139.47 83.5083.9242.78 82.39 83.50 84.80 85.71                                                                                                                                                          55.08 57.55                                                                                                                                                                                                         75.07 70.1271.9670.9969.9473.1372.8373.1470.4473.2658.9069.6461.5171.1663.1643.39 of                                        of                          MENTION       of                             of
     Qlty. GM                                             Qlty. GM           ■ ■                               Qlty. GM                                             Qlty. GM
      GB                           GB        ■ ■                           GB                           GB
TestMention NB ■ ■                     TestMention NB ■ ■ ■ ■ ■ ■                                             TestMention NB ■ ■                     TestMention NB ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■                                                                                                                                                                                                                                                                                                                                                                           Performance   G  ■    Performance      G    ■ ■ ■  ■  ■    Performance      G  ■    Performance      G

                  A ■ ■ ■    ■  ■                                                                                                                                                                                                            Syntax A ■   26:                    Syntax A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ 27:      Syntax A ■   24:                    Syntax                                           25:



                  A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■                                                                                                                                                                                                                               TrainSyntax A ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  Table                                             A ■ ■  TableTrainSyntax GA ■ ■  Table            TrainSyntax G                          G             G                                                                        Table            TrainSyntax

           Participant                                                                                                                   Participant                                                                                                                                                                                                                                   Participant                                                                                                                   Participant      best                xiong xiong                             chenyuanxiong chenyuanxiong chen chen chen chen                                    xiong xiong                             2011  fernandesmartschatbj¨orkelundchangchenstamborgchunyangyuanshouxuuryupinayangxinxinzhekovali





                                   36
<a id="page-37"></a>

### PDF 第 37 页

system in the CoNLL-2011 task was a completely            it can be useful, though not all systems that at-
rule-based one, modiﬁed version of the same system       tempted this achieved a beneﬁt, and has to be
used by shou and xiong ranked close to 10.  This       done carefully.
does indicate that a hybrid approach has some ad-
                                              • It is noteworthy that systems did not seem tovantage over a purely rule-based system. Improve-
                                                        attempt the kind of joint inference that couldment seems to be mostly owing to higher precision
                                             make use of the full potential of various lay-in mention detection, MUC, BCUBED, and higher re-
                                                              ers available in OntoNotes, but this could wellcall in CEAFe                                                    have been owing to the limited time available
9  Conclusions                                           for the shared task.
  In this paper we described the anaphoric coref-
                                              • We had expected to see more attention paid toerence information and other layers of annotation
                                                         event coreference, which is a novel feature inin the OntoNotes corpus, over three languages —
                                                                   this data, but again, given the time constraintsEnglish, Chinese and Arabic — and presented the
                                                  and given that events represent only a smallresults from an evaluation on learning such unre-
                                                           portion of the total, it is not surprising that moststricted entities and events in text. The following
                                                     systems chose not to focus on it.represents our conclusions on reviewing the results:
                                              • Scoring coreference seems to remain a signif-
  • Most top performing systems used a hybrid-                                                              icant challenge. There does not seem to be an
    approach combining rule-based strategies with                                                            objective way to establish one metric in pref-
    machine learning.  Rule-based approach does                                                        erence to another in the absence of a speciﬁc
    seem to bring a system to a close-to-best per-                                                              application.  On the other hand, the system
    formance region. The most signiﬁcant advan-                                                        rankings do not seem terribly sensitive to the
     tage of the rule-based approach seems to be                                                               particular metric chosen.  It is interesting that
     that it captures most conﬁdent links before con-                                                           the CEAFe metric — which tries to capture the     sidering less conﬁdent ones. Discourse infor-                                                    goodness of the entities in the output — seem
    mation when present is quite helpful to disam-                                           much lower than the other metric, though it is
     biguate pronominal mentions. Using informa-                                                         not clear whether that means that our systems
     tion from appositives and copular constructions                                                           are doing a poor job of creating coherent en-
    seems beneﬁcial to bridge across various lex-                                                                           tities or whether that metric is just especially
     icalized mentions.   It is not clear how much                                                            harsh.
    more can be gained using further strategies.
    The features for coreference prediction are cer-
     tainly more complex than for many other lan-                                       Acknowledgments
    guage processing tasks, which makes it more                                   We gratefully acknowledge the support of the     challenging to generate effective feature com-                                               Defense  Advanced  Research  Projects  Agency     binations.                                         (DARPA/IPTO)   under   the  GALE   program,
  • Most top performing systems did signiﬁcant                                  DARPA/CMO Contract No.  HR0011-06-C-0022.
     feature engineering – expecially a heavy use of                                  We would like to thank all the participants. Without
     lexicalized features, which was possible given                                                            their hard work, patience and perseverance this eval-
     the size of the corpus, and performed feature                                                     uation would not have been a success. We would
     selection.                                                       also like to thank the Linguistic Data Consortium
  • It might be possible that the Chinese accuracy   for making the OntoNotes v5.0 corpus freely and
     with gold boundaries and mentions is better be-   timely available in training/development/test sets
     cause the distribution of mentions across the   to the participants.  Emili Sapena, who graciously
     various genres is different, and if there are more   allowed the use of his scorer implementation. Hwee
    mentions in better scoring genres, then the per-   Tou Ng and his student Zhi Zhong for training
    formance would improve overall.                 the word  sense models and  providing  outputs
  • Gold parse during testing does seem to help   for the training/development and test sets.  Slav
     quite a bit. Gold boundaries are not of much   Petrov and Dan Klein for letting us use their parser.
     signiﬁcance for English and Arabic, but seem   Additionally, we are indebted to Slav for his help
     to be very useful for Chinese. The reason prob-   in retraining the parser for Arabic.   Alessandro
     ably has some roots in the parser performance   Moschitti and Olga Uryupina have been partially
    gap for Chinese.                             funded by  the European Community’s Seventh
  • It does seem that collecting information about   Framework Programme (FP7/2007-2013) under the
    an entity by merging information across the   grant number 288024 (LIMOSINE).
     various attributes of the mentions that comprise

                                        37
<a id="page-38"></a>

### PDF 第 38 页

References                                               Association for Computational Linguistics (ACL), Ann
Olga Babko-Malaya, Ann Bies, Ann Taylor, Szuting Yi,      Arbor, MI, June.
  Martha Palmer, Mitch Marcus, Seth Kulick, and Li-   Nancy Chinchor and Beth Sundheim.  2003.  Message
   bin Shen. 2006. Issues in synchronizing the English      understanding conference (MUC) 6. In LDC2003T13.
   treebank and propbank. In Workshop on Frontiers in   Nancy Chinchor. 2001. Message understanding confer-
   Linguistically Annotated Corpora 2006, July.             ence (MUC) 7. In LDC2001T02.
Amit Bagga and Breck Baldwin. 1998. Algorithms for   Aron Culotta, Michael Wick, Robert Hall, and Andrew
   Scoring Coreference Chains.  In The First Interna-     McCallum. 2007. First-order probabilistic models for
   tional Conference on Language Resources and Eval-      coreference resolution. In HLT/NAACL, pages 81–88.
   uation Workshop on Linguistics Coreference, pages   Pascal Denis and Jason Baldridge.  2007.   Joint de-
  563–566.                                                 termination of anaphoricity and coreference resolu-
Elizabeth Baran and Nianwen Xue.  2011.  Singular or       tion using integer programming.  In Proceedings of
   Plural? Exploiting Parallel Corpora for Chinese Num-     HLT/NAACL.
   ber Prediction. In Proceedings of Machine Translation   Pascal Denis and Jason Baldridge.  2009.  Global joint
  Summit XIII.                                        models for coreference resolution and named entity
Shane Bergsma and Dekang Lin. 2006. Bootstrapping       classiﬁcation.  Procesamiento del Lenguaje Natural,
   path-based pronoun resolution. In Proceedings of the      (42):87–96.
   21st International Conference on Computational Lin-   George Doddington, Alexis Mitchell, Mark Przybocki,
   guistics and 44th Annual Meeting of the Association     Lance Ramshaw,  Stephanie  Strassel,  and  Ralph
   for Computational Linguistics, pages 33–40, Sydney,      Weischedel. 2004. The automatic content extraction
   Australia, July.                                   (ACE) program-tasks, data, and evaluation.  In Pro-
Jie Cai and Michael Strube.  2010.  Evaluation metrics      ceedings of LREC.
   for end-to-end coreference resolution systems. In Pro-   Charles Fillmore, Christopher Johnson, and Miriam R. L.
   ceedings of the 11th Annual Meeting of the Special In-      Petruck.  2003. Background to FrameNet.  Interna-
   terest Group on Discourse and Dialogue, SIGDIAL       tional Journal of Lexicography, 16(3).
   ’10, pages 28–36.                              Ryan Gabbard. 2010. Null Element Restoration. Ph.D.
Jie Cai, Eva Mujdricza-Maydt, and Michael Strube.       thesis, University of Pennsylvania.
  2011a. Unrestricted coreference resolution via global   Aria Haghighi and Dan Klein. 2010. Coreference reso-
  hypergraph partitioning.  In Proceedings of the Fif-       lution in a modular, entity-centered model. In Human
   teenth Conference on Computational Natural Lan-     Language Technologies: The 2010 Annual Conference
  guage Learning: Shared Task, pages 56–60, Portland,       of the North American Chapter of the Association for
  Oregon, USA, June. Association for Computational      Computational Linguistics, pages 385–393, Los An-
   Linguistics.                                                   geles, California, June.
Shu Cai, David Chiang, and Yoav Goldberg.  2011b.   Jan  Hajiˇc,  Massimiliano Ciaramita,  Richard Johans-
  Language-independent parsing with empty elements.      son, Daisuke Kawahara, Maria Ant`onia Mart´ı, Llu´ıs
   In Proceedings of the 49th Annual Meeting of the As-      M`arquez, Adam Meyers, Joakim Nivre, Sebastian
   sociation for Computational Linguistics: Human Lan-       Pad´o, Jan ˇStˇep´anek, Pavel Straˇn´ak, Mihai Surdeanu,
  guage Technologies, pages 212–216, Portland, Ore-     Nianwen Xue, and Yi Zhang.  2009.  The CoNLL-
   gon, USA, June. Association for Computational Lin-     2009 shared task: Syntactic and semantic dependen-
   guistics.                                                      cies in multiple languages. In Proceedings of the Thir-
Wendy W Chapman,  Prakash M Nadkarni,  Lynette      teenth Conference on Computational Natural Lan-
  Hirschman,  Leonard W  D’Avolio,  Guergana K     guage Learning (CoNLL 2009): Shared Task, pages
   Savova, and Ozlem Uzuner. 2011. Overcoming bar-      1–18, Boulder, Colorado, June.
   riers to NLP for clinical text: the role of shared tasks   Sanda M. Harabagiu, Razvan C. Bunescu, and Steven J.
  and the need for additional creative solutions. Journal      Maiorano.  2001.  Text and knowledge mining for
   of American Medical Informatics Association, 18(5),      coreference resolution. In NAACL.
   September.                                           Lynette Hirschman and Nancy Chinchor. 1997. Corefer-
Eugene Charniak and Mark Johnson. 2001. Edit detec-      ence task deﬁnition (v3.0, 13 jul 97). In Proceedings
   tion and parsing for transcribed speech.  In Proceed-       of the Seventh Message Understanding Conference.
   ings of the Second Meeting of North American Chapter   Lynette Hirschman, Patricia Robinson, John Burger, and
   of the Association of Computational Linguistics, June.     Marc Vilain. 1998. Automating coreference: The role
Eugene Charniak and Mark Johnson. 2005. Coarse-to-      of annotated training data.  In Proceedings of AAAI
  ﬁne n-best parsing and maxent discriminative rerank-      Spring Symposium on Applying Machine Learning to
   ing. In Proceedings of the 43rd Annual Meeting of the      Discourse Processing.

                                        38
<a id="page-39"></a>

### PDF 第 39 页

Eduard Hovy, Mitchell Marcus, Martha Palmer, Lance   David S. Pallett. 2002. The role of the National Insti-
  Ramshaw, and Ralph Weischedel. 2006. OntoNotes:       tute of Standards and Technology in DARPA’s Broad-
  The 90% solution.  In Proceedings of HLT/NAACL,       cast News continuous speech recognition research pro-
  pages 57–60, New York City, USA, June. Association      gram. Speech Communication, 37(1-2), May.
   for Computational Linguistics.                      Martha Palmer,  Daniel Gildea, and Paul Kingsbury.
Karin Kipper, Anna Korhonen,  Neville Ryant,  and      2005. The Proposition Bank: An annotated corpus of
  Martha Palmer. 2000. A large-scale classiﬁcation of      semantic roles. Computational Linguistics, 31(1):71–
   english verbs. Language Resources and Evaluation,      106.
   42(1):21 – 40.                                    Martha Palmer, Hoa Trang Dang, and Christiane Fell-
Heeyoung Lee, Yves Peirsman, Angel Chang, Nathanael     baum. 2007. Making ﬁne-grained and coarse-grained
  Chambers, Mihai Surdeanu, and Dan Jurafsky. 2011.      sense distinctions, both manually and automatically.
   Stanford’s multi-pass sieve coreference resolution sys-                                                  Martha Palmer, Olga Babko-Malaya, Ann Bies, Mona
  tem at the conll-2011 shared task.  In Proceedings                                                          Diab, Mohammed Maamouri, Aous Mansouri, and
   of the Fifteenth Conference on Computational Natu-     Wajdi Zaghouani. 2008. A pilot arabic propbank. In
   ral Language Learning: Shared Task, pages 28–34,      Proceedings of the International Conference on Lan-
   Portland, Oregon, USA, June. Association for Com-     guage Resources and Evaluation (LREC), Marrakech,
   putational Linguistics.                                                    Morocco, May 28-30.
Xiaoqiang Luo. 2005. On coreference resolution perfor-                                                  Rebecca Passonneau.  2004.  Computing reliability for
  mance metrics.  In Proceedings of Human Language                                                            coreference annotation. In Proceedings of LREC.
  Technology Conference and Conference on Empirical
                                                       Slav Petrov and Dan Klein. 2007. Improved Inferencing
  Methods in Natural Language Processing, pages 25–                                                                 for Unlexicalized Parsing. In Proc of HLT-NAACL.
   32, Vancouver, British Columbia, Canada, October.
                                              Massimo Poesio and Ron Artstein. 2005. The reliabilityMohamed Maamouri and Ann Bies. 2004. Developing
                                                             of anaphoric annotation, reconsidered: Taking ambi-  an arabic treebank: Methods, guidelines, procedures,
                                                              guity into account. In Proceedings of the Workshop on  and tools. In Ali Farghaly and Karine Megerdoomian,
                                                              Frontiers in Corpus Annotations II: Pie in the Sky.   editors, COLING 2004 Computational Approaches to
                                              Massimo Poesio.  2004.  The mate/gnome scheme for  Arabic Script-based Languages, pages 2–9, Geneva,
                                                          anaphoric annotation, revisited.   In Proceedings of   Switzerland, August 28th. COLING.
                                                  SIGDIAL.Mitchell P. Marcus, Beatrice Santorini, and Mary Ann
                                                Simone Paolo Ponzetto and Massimo Poesio.   2009.   Marcinkiewicz. 1993. Building a large annotated cor-
                                                                  State-of-the-art nlp approaches to coreference resolu-  pus of English: The Penn treebank.  Computational
                                                                     tion: Theory and practical recipes.  In Tutorial Ab-   Linguistics, 19(2):313–330, June.
                                                                   stracts of ACL-IJCNLP 2009, page 6, Suntec, Singa-Andrew McCallum and Ben Wellner. 2004. Conditional
                                                              pore, August.  models of identity uncertainty with application to noun
   coreference. In Advances in Neural Information Pro-   Simone Paolo Ponzetto and Michael Strube. 2005. Se-
   cessing Systems (NIPS).                                mantic role labeling for coreference resolution.  In
Joseph McCarthy and Wendy Lehnert. 1995. Using de-     Companion Volume of the Proceedings of the 11th
   cision trees for coreference resolution. In Proceedings     Meeting of the European Chapter of the Associa-
   of the Fourteenth International Conference on Artiﬁ-       tion for Computational Linguistics, pages 143–146,
   cial Intelligence, pages 1050–1055.                        Trento, Italy, April.
Thomas S. Morton. 2000. Coreference for nlp applica-   Simone Paolo Ponzetto and Michael Strube.   2006.
   tions. In Proceedings of the 38th Annual Meeting of      Exploiting  semantic  role  labeling,  WordNet  and
   the Association for Computational Linguistics, Octo-      Wikipedia for coreference resolution. In Proceedings
   ber.                                                          of the HLT/NAACL, pages 192–199, New York City,
Eugene W. Myers.  1986. An O(ND) difference algo-      N.Y., June.
   rithm and its variations.  Algorithmica, 1(2):251—-   Sameer  Pradhan,  Kadri  Hacioglu,  Valerie  Krugler,
   266.                                         Wayne Ward, James Martin, and Dan Jurafsky. 2005.
Vincent Ng.  2007.  Shallow semantics for coreference      Support vector learning for semantic argument classi-
   resolution. In Proceedings of the IJCAI.                    ﬁcation. Machine Learning Journal, 60(1):11–39.
Vincent Ng. 2010. Supervised noun phrase coreference   Sameer Pradhan, Eduard Hovy, Mitchell Marcus, Martha
   research: The ﬁrst ﬁfteen years. In Proceedings of the      Palmer,  Lance Ramshaw,  and Ralph Weischedel.
   48th Annual Meeting of the Association for Compu-      2007a.  OntoNotes: A Uniﬁed Relational Semantic
   tational Linguistics, pages 1396–1411, Uppsala, Swe-      Representation.   International Journal of Semantic
   den, July.                                           Computing, 1(4):405–419.

                                        39
<a id="page-40"></a>

### PDF 第 40 页

Sameer Pradhan, Lance Ramshaw, Ralph Weischedel,       electronic medical records. Journal of American Med-
   Jessica MacBride, and Linnea Micciulla.   2007b.       ical Informatics Association, 19(5), September.
   Unrestricted Coreference:  Indentifying Entities and   Yannick Versley, Simone Paolo Ponzetto, Massimo Poe-
  Events  in OntoNotes.   In  in Proceedings  of the       sio,  Vladimir Eidelman, Alan  Jern,  Jason Smith,
  IEEE International Conference on Semantic Comput-      Xiaofeng Yang, and Alessandro Moschitti.   2008.
   ing (ICSC), September 17-19.                     BART: A modular toolkit for coreference resolution.
Karthik Raghunathan, Heeyoung Lee, Sudarshan Ran-      In Proceedings of the 6th International Conference on
   garajan, Nate Chambers, Mihai Surdeanu, Dan Juraf-     Language Resources and Evaluation, Marrakech, Mo-
   sky, and Christopher Manning.  2010. A multi-pass      rocco, May.
   sieve for coreference resolution.  In Proceedings of   Yannick Versley. 2007. Antecedent selection techniques
   the 2010 Conference on Empirical Methods in Natu-       for high-recall coreference resolution. In Proceedings
   ral Language Processing, pages 492–501, Cambridge,       of the 2007 Joint Conference on Empirical Methods
  MA, October. Association for Computational Linguis-       in Natural Language Processing and Computational
   tics.                                                   Natural Language Learning (EMNLP-CoNLL).
Altaf Rahman and Vincent Ng. 2009. Supervised mod-   Marc Vilain, John Burger, John Aberdeen, Dennis Con-
   els for coreference resolution.  In Proceedings of the       nolly, and Lynette Hirschman. 1995. A model theo-
  2009 Conference on Empirical Methods in Natural       retic coreference scoring scheme.  In Proceedings of
  Language Processing, pages 968–977, Singapore, Au-      the Sixth Message Undersatnding Conference (MUC-
   gust. Association for Computational Linguistics.             6), pages 45–52.
William M. Rand. 1971. Objective criteria for the evalu-   Ralph Weischedel and Ada Brunstein. 2005. BBN pro-
   ation of clustering methods. Journal of the American     noun coreference and entity type corpus LDC catalog
   Statistical Association, 66(336).                              no.: LDC2005T33. BBN Technologies.
                                                   Ralph  Weischedel,  Eduard Hovy,  Mitchell Marcus,Marta Recasens and Eduard Hovy.  2011.  Blanc: Im-
                                                     Martha  Palmer,  Robert  Belvin,  Sameer  Pradhan,  plementing the rand index for coreference evaluation.
                                                    Lance  Ramshaw,  and  Nianwen  Xue.     2011.   Natural Language Engineering.
                                                       OntoNotes: A large training corpus for enhanced pro-Marta  Recasens,   Llu´ıs  M`arquez,   Emili  Sapena,
                                                               cessing.  In Joseph Olive, Caitlin Christianson, and  M.  Ant`onia   Mart´ı,  Mariona   Taul´e,  V´eronique
                                                      John McCary, editors, Handbook of Natural Language   Hoste, Massimo Poesio, and Yannick Versley. 2010.
  Semeval-2010  task  1:   Coreference  resolution  in      Processing and Machine Translation: DARPA Global
                                                   Autonomous Language Exploitation. Springer.   multiple languages. In Proceedings of the 5th Interna-
                                               Nianwen Xue and Martha Palmer. 2009. Adding seman-   tional Workshop on Semantic Evaluation, pages 1–8,
                                                                            tic roles to the Chinese Treebank. Natural Language   Uppsala, Sweden, July.
                                                           Engineering, 15(1):143–172.Wee Meng Soon, Hwee Tou Ng, and Daniel Chung Yong
                                               Nianwen Xue, Fei Xia, Fu dong Chiou, and Martha  Lim. 2001. A machine learning approach to corefer-
                                                           Palmer. 2005. The Penn Chinese TreeBank: Phrase  ence resolution of noun phrase. Computational Lin-
                                                              Structure Annotation of a Large Corpus. Natural Lan-   guistics, 27(4):521–544.
                                                    guage Engineering, 11(2):207–238.Veselin Stoyanov, Nathan Gilbert, Claire Cardie, and
                                               Nianwen Xue.   2008.   Labeling Chinese Predicates   Ellen Riloff. 2009. Conundrums in noun phrase coref-
                                                         with Semantic Roles.   Computational Linguistics,   erence resolution: Making sense of the state-of-the-
                                                         34(2):225–255.   art. In Proceedings of the Joint Conference of the 47th                                                    Yaqin Yang and Nianwen Xue.  2010.  Chasing the
  Annual Meeting of the ACL and the 4th International
                                                              ghost: recovering empty categories in the chinese tree-
   Joint Conference on Natural Language Processing of
                                                          bank. In Proceedings of Proceedings of the 23rd In-
   the AFNLP, pages 656–664, Suntec, Singapore, Au-
                                                               ternational Conference on Computational Linguistics
   gust. Association for Computational Linguistics.                                                  (COLING), Beijing, China.
Mihai Surdeanu,  Richard Johansson, Adam Meyers,                                                   Wajdi Zaghouani, Mona Diab, Aous Mansouri, Sameer
   Llu´ıs M`arquez, and Joakim Nivre. 2008. The CoNLL                                                         Pradhan, and Martha Palmer. 2010. The revised ara-
  2008 shared task on joint parsing of syntactic and se-                                                              bic propbank. In Proceedings of the Fourth Linguistic
  mantic dependencies.  In CoNLL 2008: Proceedings                                                         Annotation Workshop, pages 222–226, Uppsala, Swe-
   of the Twelfth Conference on Computational Natu-                                                            den, July.
   ral Language Learning, pages 159–177, Manchester,                                                    Zhi Zhong and Hwee Tou Ng.  2010.  It makes sense:
  England, August.                                   A wide-coverage word sense disambiguation system
Ozlem Uzuner, Andreea Bodnari, Shuying Shen, Tyler                                                                 for free text. In Proceedings of the ACL 2010 System
   Forbush, John Pestian, and Brett R South. 2012. Eval-                                                         Demonstrations, pages 78–83, Uppsala, Sweden.
   uating the state of the art in coreference resolution for

                                        40
