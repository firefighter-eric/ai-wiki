# Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co

- Source PDF: `raw/pdf/Dumas, Organisciak, Doherty - 2020 - Measuring Divergent Thinking Originality With Human Raters and Text-Mining Models A Psychometric Co.pdf`
- Source SHA256: `bdf127a7b91bdc9674840a7644128498c3849b97d8a33801e22573ba903d072e`
- Generated from: `scripts/extract_pdf_text.py`

- Extraction: `pymupdf-pages-v2` (sorted text, page anchors; tables/formulas/figures require review)

## Extracted Text

<a id="source-section-0"></a>

<a id="page-1"></a>

### PDF 第 1 页

Psychology of Aesthetics, Creativity,
and the Arts

Measuring Divergent Thinking Originality With Human
Raters and Text-Mining Models: A Psychometric
Comparison of Methods
Denis Dumas, Peter Organisciak, and Michael Doherty
Online First Publication, July 23, 2020. http://dx.doi.org/10.1037/aca0000319


CITATION
Dumas, D., Organisciak, P., & Doherty, M. (2020, July 23). Measuring Divergent Thinking Originality
With Human Raters and Text-Mining Models: A Psychometric Comparison of Methods. Psychology
of Aesthetics, Creativity, and the Arts. Advance online publication.
http://dx.doi.org/10.1037/aca0000319
<a id="page-2"></a>

### PDF 第 2 页

Psychology of Aesthetics, Creativity, and the Arts


         © 2020 American Psychological Association                                                                                                                       2020, Vol. 2, No. 999, 000
              ISSN: 1931-3896                                                                                                                                                http://dx.doi.org/10.1037/aca0000319


      Measuring Divergent Thinking Originality With Human Raters and Text-
             Mining Models: A Psychometric Comparison of Methods




                Denis Dumas and Peter Organisciak                           Michael Doherty
                                University of Denver                                 Actor’s Equity Association, New York, New York



                                 Within creativity research, interest and capability in utilizing text-mining models to quantify the          broadly.
                                      Originality of participant responses to Divergent Thinking tasks has risen sharply over the last decade,
                                  with many extant studies fruitfully using such methods to uncover substantive patterns among creativity-publishers.
                                      relevant constructs. However, no systematic psychometric investigation of the reliability and validity of
                                 human-rated Originality scores, and scores from various freely available text-mining systems, exists inallied               disseminated                                the literature. Here we conduct such an investigation with the Alternate Uses Task. We demonstrate that,
its
   be                                despite their inherent subjectivity, human-rated Originality scores displayed the highest reliability at both
of
   to                                the composite and latent factor levels. However, the text-mining system GloVe 840B was highly capable
one    not                               of approximating human-rated scores both in its measurement properties and its correlations to various
or is                                  creativity-related criteria including ideational Fluency, Elaboration, Openness, Intellect, and self-reported
    and                               Creative Activities. We conclude that, in conjunction with other salient indicators of creative potential,
                                    text-mining models (and especially the GloVe 840B system) are capable of supporting reliable and valid
     user                                inferences about Divergent Thinking. We offer an open-access module for researchers to apply theseAssociation                             methods to their own data via our laboratory website (https://openscoring.du.edu/).

                                Keywords: creativity, divergent thinking, psychometrics, reliability, text-mining models             individual
    the                             Supplemental materials: http://dx.doi.org/10.1037/aca0000319.suppPsychological of
American use        Divergent thinking (DT)—or the human mental ability to gen-     task responses are closely considered, and at times psychometri-the  personal       erate multiple original ideas in response to a given problem or     cally examined, by creativity researchers (e.g., Dumas & Dunbar,
by the     prompt (Acar & Runco, 2019; Forthmann, Wilken, Doebler, &    2014; Kuhn & Holling, 2009). However, despite the broad use of
    for      Holling, 2019)—has long been a centrally important construct     the AUT in creativity research, the extant psychometric under-
        solely      underHocevar,investigation1980). Today,withinDTthetaskscreativityare byresearchfar theliteraturemost utilized(e.g.,     standingdevelopedofas participantin other areasscoresof psychologyfrom the (e.g.,AUTclinicalis simplypersonalitynot ascopyrighted        measures in creativity research (Plucker & Makel, 2010; Reiter-     assessment; Marek & Ben-Porath, 2017), for a variety of reasons.
is        Palmon, Forthmann, & Barbot, 2019), with the Alternate Uses                                                                                  In our view, one principal reason why DT measurement  is          intended     Task (AUT; Guilford, 1967; Hudson, 1968; Torrance, 1972) being
   is                                                              somewhat underdeveloped is because DT tasks fundamentally rely
           the chief among them. Therefore, procedures and methods that aredocument                                                                 on open-ended participant responses, whereas the vast majority of          used to calculate participant scores from their AUT or other DT         article                                                                      psychometric modeling frameworks that can be used to evaluateThis
     This                                                                             the reliability or internal validity of scoring models were designed                                                                                      for close-ended assessment  (e.g., item response theory; Lord,
                                                                             2012). In fact, the AUT is not only open-ended in its response
                                                                           format (i.e., participants must verbally respond or write-in their
             Denis Dumas and X Peter Organisciak, Department of Research Meth-
                                                                             responses rather than selecting a response-option)  it is also  ill-           ods and Information Science, Morgridge College of Education, University
            of Denver; Michael Doherty, Actor’s Equity Association, New York, New     structured (i.e., participants can differ substantially on the number
           York.                                                                of responses they supply and the length of those responses): an
              This research was supported financially through a research seed-grant    assessment format that stumps most currently dominant psycho-
           from the University of Denver’s Morgridge College of Education. We     metric scoring methods. One approach to the scoring of open-
            thank Amanda Strickland, Megan Solberg, and Danielle Francisco Albu-                                                                      ended divergent thinking tasks that focuses on taking a relatively
           querque Vasques for their critical assistance in coding the responses.
                                                                                 rapid snapshot of creativity  is a subjective scoring procedure
             Correspondence concerning this article should be addressed to Denis
          Dumas, Department of Research Methods and Information Science, Mor-     introduced by Silvia and colleagues (2008). However, the search
            gridge College of Education, University of Denver, 1999 East Evans     for more objective performance measurement strategies for diver-
           Avenue, Denver, CO 80210. E-mail: denis.dumas@du.edu                 gent thinking tasks—especially measurement strategies that can be

                                                                        1
<a id="page-3"></a>

### PDF 第 3 页

2                                 DUMAS, ORGANISCIAK, AND DOHERTY


          used as part of an automatic scoring system—has been more     their relationship and represent each of those words similarly
           elusive (Dumas & Runco, 2018).                                    across the latent dimensions of the model. This has the effect of
              Fortunately, relatively recent advances in text-mining method-     accounting for similar words and synonyms. Even though the
          ology  (e.g., Hirschberg & Manning, 2015) are offering some    words in “the dog barked” and “the puppy barked” are different,
           solutions for the measurement of psychological attributes from     the contextual difference in not great, and they will appropriately
          open-ended and ill-structured assessment data. With the advent of    be understood as similar in an LSA representation of those state-
          a relatively new interdisciplinary area of research that combines    ments (Günther, Dudschig, & Kaup, 2015). The substantive effect
           text-mining and psychometric methods—sometimes termed com-     of this modeling of latent dimensions is that words have quantifi-
           putational psychometrics (von Davier, 2017)—the reliable quan-     able distances between themselves in the trained model, and how
             tification of human mental attributes that are too complex to be     close or far two words are from each other tends to align with
           assessed via close-ended test items (e.g., DT) is becoming more    humans’ perceptions of whether words are semantically similar or
           possible than ever. In the next sections of this article, we first     dissimilar (Landauer & Dumais, 1997).
            briefly overview areas in which computational psychometrics has       This relatively nuanced representation of the semantic structure
         opened important doors in psychological research and applied     of language through text-mining models has allowed psychome-
            testing contexts, and then move to a detailed review of the existing     tricians to begin to objectively and automatically quantify such          broadly.
          computational psychometric applications in the creativity research    complex psychological constructs as writing ability (i.e., via essaypublishers.          literature.                                                     prompts and automatic essay scoring systems; Foltz, Streeter,
allied                                                                Lochbaum,tual analysis& ofLandauer,patient interview2013), depressiondata; Kjell,(i.e.,Kjell,throughGarcia,the tex-&               disseminated              Computational Psychometrics for
its                                                                           Sikström, 2019), and collaborative ability (i.e., by examining the
   be                 Open-Ended Measuresof   to                                                                      semantic structure of online chats in which participants collaborate
one          For decades, social scientists engaged in basic research have     to solve an abstract problem; He, von Davier, Greiff, Steinhauer, &    not
or is      understood that a close-ended psychometric methodological per-    Borysewicz, 2017). Indeed, all of these applications are beginning
    and      spective—in which participants are administered questionnaires or     to become relatively large scale, with automatic essay scoring
              tests, and the answer-choices that participants select are used to    systems being deployed widely in standardized admissions testing,
     user       calculate their score—has some serious limitations for the study of    mental health related text-mining systems beginning to make theirAssociation
         complex human thought and behavior (e.g., Messick, 1995). For    debut in clinics, and the text-based assessment of student collab-
          example, many investigations have utilized rich sources of inter-     orative skills being administered to students all around the world             individual      view,  observation,  essay,  or  otherwise  open-ended  and   ill-     as  part of the Program  for  International Student Assessment
    the      structured data, which is typically scored through the application    (PISA) tests.Psychological of      of human-raters who judge the degree to which those data indicate
    use      the presence of a given psychological construct within a partici-     Computational Psychometrics in Creativity Research
           pant (e.g., Bråten, Ferguson, Strømsø, & Anmarkrud, 2014). How-
                                                                             Since the theorizing of Mednick (1962), creativity has oftenAmerican         ever, the use of such human-raters is costly both in terms of time
                                                                      been considered from the viewpoint of associative distance. Fromthe  personal     and money, and the reliability and validity of human-rated partic-                                                                                            this perspective, the Originality of a given response is conceptu-
by the      ipant scores  is subject to measurement error from the raters’
                                                                                    alized as arising from its distal-relatedness from the context or
    for       implicit beliefs and biases. The subjective nature of such human-
                                                                     prompt from which it arose, such that a response that would be
           rated scores has often been criticized in the literature (Gwet, 2014),
                                                                                     typically associated with a given prompt or context would be        solely      although, in the case of open-ended and ill-structured data sources,
                                                                             considered to be low in terms of Originality, whereas a responsecopyrighted       no alternative has historically existed.
is                                                                                      that is more unusual within a given context would be considered to
           However, in the last decade, major advances have been made in
                                                                      be more Original. Importantly, the theoretical conceptualization of          intended      devising methods for the objective  (i.e., not human-rated) and
   is                                                                              Originality as the associative distance between a prompt and the
          automatic (i.e., computer-generated) scoring of open-ended psy-document                                                                          response a participant generates can be operationalized (at least for           chological data sources, at least those that are textual or verbal in         article                                                                           verbal or textual DT responses such as from the AUT) as theThis         format. These computational psychometric approaches are based
     This     on text-mining models that are trained on a massive corpus of    semantic distance between a given response and the AUT prompt                                                                                                  (e.g., “Brick”) from which it arose (Beaty, Silvia, Nusbaum, Jauk,
           extant text (e.g., a library of digitized books; Crossley, Dascalu, &
                                           & Benedek, 2014). This theoretical pairing of the associative
         McNamara, 2017) to represent the semantic meaning of words in
                                                                                 distance among ideas and the semantic distance among the words
           the context in which they are used. For example, the most popular
                                                                         used to express these ideas (i.e., the responses) may not be per-
           of these text-mining models in the psychological literature today is
                                                                                        fectly one-to-one (cf. historic “thought and language” debates in
           Latent Semantic Analysis (LSA; Landauer,  Foltz, & Laham,
                                                                           psychology;  cf., Piatelli-Palmarini, 1980), but  it appears close
           1998), which is a dimensionality reduction technique that seeks to
                                                                    enough to justify the application of computational psychometric
          reduce a sparse-matrix of document-term counts, where each doc-
                                                                          approaches (e.g., LSA) to DT data to quantify the semantic dis-
         ument is represented by the counts of all the words in the model
                                                                                tances among the prompts and responses.
           vocabulary, to a much smaller representation of documents as a
         few hundred features. This is done by finding co-occurrence pat-
                                                               Latent Semantic Analysis and Divergent Thinking
            terns between all the words in a document. For example, rather
           than representing the words dog, puppy, and canine all as inde-       In the creativity research literature, the text-mining model that is
          pendent quantitative counts, an LSA trained model would learn    most typically utilized is LSA (Acar & Runco, 2019). LSA has
<a id="page-4"></a>

### PDF 第 4 页

MEASURING ORIGINALITY                                       3


          been an appropriate and useful method for the quantification of     Forster and Dunbar (2009) presented the first formal evidence of
           Originality because the process of training an LSA system  is     the appropriateness of LSA based Originality scores. In this work,
            effective at preserving the linearity of the relations among words,     they showed that LSA Originality scores were capable of discrim-
          so that the semantic distances between them are directly compa-     inating between groups of participants who received different
           rable by studying the factorization matrix of words by latent     directions to the AUT (i.e., creative responses and common re-
          dimensions. Using these matrices, the latent dimensions in an LSA     sponses), and  that LSA based  Originality scores were more
         model can be used as coordinates in a geometrically represented     strongly predictive of human-rated Originality than were other
           space, and the cosine of the angle between the word-vectors can be   common DT metrics such as Fluency and Elaboration.
            interpreted as the semantic or associative distance among words       Following this first foray into LSA based Originality scoring,
           (Deerwester, Dumais, Furnas, Landauer, & Harshman, 1990). As    Green and colleagues (e.g., Green, Kraemer, Fugelsang, Gray, &
          an example with the AUT, if the prompt was “fork,” the response    Dunbar, 2010) used LSA to examine the link between participant
           “eat pasta” would result in a vector that has an acute angle with the     relational reasoning abilities and divergent thinking. In addition to
           vector for “fork.” In Contrast, the response “conduct electricity”      this work, Dumas and Dunbar (2014) published the first psycho-
         would result in a vector that has a wider angle from the initial     metric investigation of LSA based Originality scores and their
         prompt vector. Please see Figure 1 for a visualization of the     relation to Fluency scores from a latent variable perspective. This          broadly.
          geometric relations among AUT prompt and responses in LSA. To     study found that a confirmatory factor analysis (CFA) model thatpublishers.         calculate the Originality scores for these AUT responses, the     included both LSA based Originality scores and Fluency counts
allied         cosinedistanceofbetweenthe anglethewouldpromptbe andcalculatedresponse,to representand then thatthe semanticsemantic     acrossvery closely10 AUTandpromptsachievedwasa highcapabledegreeof offittingreliabilitythe observedfor bothdatathe               disseminated
its         distance would be subtracted from 1 to yield an Originality score    Fluency and Originality latent factors. In this way, LSA Original-   be
of   to       for each response.                                                            ity scores have demonstrated discriminant validity from Fluency
one          Following this general methodological paradigm, a number of     scores, and in another analysis, were shown to demonstrate a high    not
or is      studies both of the measurement-related functioning of LSA based    degree of reliability even after the variance explained by Fluency
    and      Originality scores (e.g., Dumas & Runco, 2018; Forthmann, Oye-    was partialed out (Dumas & Runco, 2018).
           bade, Ojo, Günther, & Holling, 2019; Heinen & Johnson, 2018;      Over the next few years, a solid handful of substantive appli-
     user      Prabhakaran, Green, & Gray, 2014), as well the application of     cations of LSA Originality scores appeared in the creativity liter-Association
       LSA Originality scores to answering substantive research ques-     ature, demonstrating that this method was capable of producing
            tions in the creativity research literature (e.g., Hass, 2017a; White     scores that allowed creativity researchers to gain insights about             individual   & Shah, 2016) have appeared. As far as we are aware, Kevin     psychological phenomena  related  to  creativity. For example,
    the     Dunbar and his students and collaborators (e.g., Dumas & Dunbar,    White and Shah (2016) showed that LSA semantic distancesPsychological of      2014; Forster & Dunbar, 2009; Green, Kraemer, Fugelsang, Gray,    among word association pairs were capable of explaining observed
    use   & Dunbar, 2012) were the first to recognize the rich possibilities    advantages of individuals with attention-deficit/hyperactivity dis-
            in the application of LSA to DT and other cognitive tasks. As such,     order (ADHD) on a DT task, leading to the hypothesis that ADHDAmerican                                                        may support DT in part because of the wider scope of semanticthe  personal                                                                               activation in those individuals. The same year, Dumas and Dunbar
by the                                                                      (2016) showed that participants’ LSA-based Originality scores
    for                                                                were significantly influenced by DT task instructions, particularly
        solely                                                          whencally creativeparticipantsindividualswere askedsuchtoastakepoets.the perspectiveThe followingof stereotypi-year, twocopyrighted                                                                          papers by Hass (2017a, 2017b) applied LSA-based Originality
is                                                                              scores to a fine-grained analysis of DT production over short spans          intended                                                                           of time. For example, Hass (2017a) showed that the LSA-based
   is                                                                      semantic similarity (the inverse of Originality) of AUT responsesdocument                                                                    were negatively correlated with human-judged creativity ratings,         articleThis                                                                  which provided evidence for the validity of LSA-based DT scores.
     This                                                                           In the same article, the semantic similarity of AUT responses
                                                                           followed a cubic trend overtime: The most highly semantically
                                                                                    similar responses to the prompt occurred first, the similarity of
                                                                             responses then decreased (or Originality increased) over the first
           Figure 1. A visual representation of the geometric relations among vec-     five or so responses, before increasing again between the fifth and
              tors arising from an Latent Semantic Analysis (LSA) analysis. These    10th responses, and after the 10th response the similarity decreased
            angles were calculated using an LSA model trained on the Touchstone    once more. In addition, Hass (2017b) then demonstrated that the
           Applied Science Associates (TASA) corpus. To calculate the Originality of                                                                                        fluid intelligence of participants significantly influenced their abil-
           each of the Alternate Uses Task (AUT) responses, the cosine of the angle
                                                                                              ity to generate semantically distant (i.e., Original) responses to the
          would be calculated to produce a semantic similarity score, and then that
                                                 AUT (although the change overtime of semantic distance was            score would be subtracted from one to generate the Originality for each
             use. The cosine distance is taken to account for document length: Because      linear, not cubic in that investigation).
           words are represented in the latent model, multiword phrases can be        Later, Dumas (2018) used LSA-based Originality scores to test
            represented as a sum of all word vectors (e.g., vec(eat)   vec(pasta)) while     the  long-standing  threshold  hypothesis  (e.g., Karwowski &
             retaining the ability to compare them with one-word phrases.               Gralewski, 2013) in the creativity literature: that intellectual ability
<a id="page-5"></a>

### PDF 第 5 页

4                                 DUMAS, ORGANISCIAK, AND DOHERTY


           supports creative ability, but only up to a point. In this study,     relations between words, more training texts generally lead to a
         LSA-based Originality scores on the AUT were able to reach a     better sense of the true semantic structure of language, and the
           strong level of scale reliability, and allowed for an analysis in   TASA corpus is relatively small by modern standards. Finally, the
         which the threshold hypothesis was supported under some condi-     type of texts trained on will affect the relations between words
            tions, but not in others. This finding, along with those of the Hass     because, say, new articles, conversational posts, and legal argu-
           (2017a, 2017b) articles, may illustrate how semantic distance can    ments all have different styles of language. Which domain of
          be used as a fruitful operationalization of Originality to address     corpora leads to semantic models that are most appropriate for
           longstanding questions in creativity research. In addition, Dumas     Originality scoring is not yet known, and such a question seems
         and Strickland (2018) applied LSA-based Originality scores to    worth exploring beyond the formal educational texts uses for the
            investigate malevolent or violent responses on the AUT, and found   TSA corpus.
            that those participants who scored more highly on their Originality       Currently, only a small minority of creativity researchers have
           also produced more violent responses on the AUT (e.g., “kill    used alternative training corpora for their LSA-based investiga-
         someone” as a use for “shovel”). Such a finding adds to the     tions. For example Forthmann and colleagues (2019) have used the
            potential evidence for the predictive validity of LSA-based Orig-    more modern English 100k corpus, which is much more generally
             inality scores, because those scores were capable of significantly    based on a web crawl of .uk domains of the Internet. However,          broadly.
         and positively predicting a theoretically relevant creativity-related    most LSA-based creativity research (e.g., Dumas & Dunbar, 2014,publishers.         construct (i.e., malevolence). Even more recently, Gray and col-    2016; Gray et al., 2019) continues to be done using the TASA
allied         leaguesdistance (2019)among usedresponsesLSA-basedto wordscoresassociationto quantifytasks. theSimilarlysemanticto     corpus,Elaborationraising aconfound.possible validityBecauserisk.participant responses to DT               disseminated
its        Hass (2017a), these researchers pointed out that the changes in     tasks can contain varying amounts of words (i.e., they can vary in   be
of   to      semantic distance over time as participants generate responses is     their Elaboration), LSA-based scoring may be adversely affected
one         informative as to their creative potential.                        by these differences. As first pointed out by Forster and Dunbar    not
or is                                                                      (2009) and technically examined by Forthmann and colleagues
    and     Limitations of LSA                                              (2019), LSA-based Originality scores at the participant level com-
                                                                 monly exhibit a substantial correlation with Elaboration, implying
     user        Since the very first applications of LSA in the psychological     that the more words a participant uses to explain their idea, theAssociation
             literature (see Landauer, McNamara, Dennis, & Kintsch, 2013 for    more LSA estimates of Originality are confounded. As a slightly
          a handbook of reviews), it has been understood that the reliability    more technical description, because LSA Originality scores are             individual     and validity of participant scores from computational psychomet-    based on vectors for the entire DT response and not just individual
    the       ric models that incorporate LSA depend on a number of factors.    words (see Figure 1), those vectors are essentially composed of thePsychological of      Like  all DT assessment, the quality of LSA-based Originality    sum of the individual word vectors for every word in the DT
    use      scores depends on a myriad of administrative and scoring choices,     response (Landauer, Laham, Rehder, & Schreiner, 1997). Because
           but some that are of particular importance in the LSA context    some words used in a response (e.g., “and” or “so”) may be veryAmerican         include: (a) the corpus from which the LSA system was trained, (b)    commonly utilized, they are not particularly discriminatory. Thesethe  personal      the particular methodological decisions for how to handle ex-    commonly used function words lower the semantic distance of the
by the      tremely common words like “and” or “is,” and (c) the way in      full DT response from the prompt, even though the core idea of the
    for     which semantic similarity or distance scores for individual DT     response may have been highly Original. Such an understanding of
        solely      responsesparticipantarelevel.aggregatedEach oftothesethe promptissues is(e.g.,now“Brick”)briefly explained.level, or the   LSAfor Elaborationmakes it clearto identifythat, going(and forward,possibly controlDT tasksfor)mustconfoundsbe scoredincopyrighted          Training  corpus.  Beginning  with  Forster  and  Dunbar’s     substantive studies. In addition, the relation between text-mining–
is        (2009) initial work, by far the most widely utilized training corpus    based Originality scores and human rated Originality scores needs          intended       for LSA in the creativity research literature has been the Touch-     to continue to be checked to ensure that the influence of Elabora-
   is      stone Applied Science Associates (TASA) corpus. This corpus was     tion does not throw-off this relation. In addition, Forthmann anddocument            originally created by Landauer and Dumais (1997) and was ini-     colleagues (2019) do offer some statistical corrections for formu-         articleThis           tially applied to the psychological research literature by Walter     lating LSA-based Originality scores that can alleviate this prob-
     This      Kintsch and his colleagues (e.g., Kintsch & Bowles, 2002). As part     lem. For instance, one way to control for the misleading effects of
           of the interdisciplinary work among these scholars, a freely acces-   common function words is “stopword lists,” which simply remove
            sible tool was made available to access this corpus, originally     (or stop) a set of words based on a known list. In this article, a
          through an Internet browser (i.e., lsa.colorado.edu) but today also     correction known as term weighting is applied to all systems under
          through the open-source software r (i.e., LSAfun; Günther et al.,     investigation, to weight different words to be more or less impact-
           2015). This corpus is composed of nearly 40 thousand educational     ful based on how discriminatory they are. Here, Inverse-document-
            texts and is meant to represent the average reading experience of    frequency (IDF) from the information science literature is used as
           the typical entering American undergraduate student (who are the     the term weighting method (Robertson & Jones, 1976), using
         most commonly recruited participants in psychology  studies).    precomputed term weights that emphasize the influence of less
         However, the TASA corpus was created in the late 1990s and, as   common words (Organisciak, 2016). Correlations with Elaboration
            far as we are aware, has not been updated since the very early     scores are also checked here as an index of discriminant validity.
          2000s, calling into serious question the capacity of this corpus to      Scoring aggregation method.  One perhaps unfortunate pat-
           continue to represent the true semantic relations of current lan-     tern within the creativity research literature is that DT assessments
          guage. Because the goal of the system  is to adequately learn     are often thought of by researchers simply as tasks and not mea-
<a id="page-6"></a>

### PDF 第 6 页

MEASURING ORIGINALITY                                       5


            sures. Although this is a subtle distinction, task is a much more    tem over others that may provide us with better information about
           general category that includes any stimuli designed to elicit a     Originality.
            certain cognitive process or behavior from participants, whereas a      Today, there are a number of other freely available text-mining
          measure requires multiple items or indicators of an underlying    systems in the information science and computer science commu-
            latent mental attribute to be aggregated, or scored, to represent a      nities that creativity researchers may recruit for their work. These
           psychologically meaningful quantity (Hedge, Powell, & Sumner,     available text-mining systems differ on the type of model they
           2018). For example, if the AUT is administered to participants, but      utilize (i.e., they do not use LSA), the text corpus they used to train
          only one or two prompts (e.g., “Book,” “Hammer”) are included,     the model, and the preprocessing and parameterization performed
           those few prompts cannot be aggregated in a way that provided     in training. For this reason, even across freely available text-
          psychometric evidence of the reliability and internal validity of the    mining systems, there is a high potential for very different psy-
            scores. In the creativity literature, observed-variable checks of the    chometric and psychological patterns to emerge in Originality
          composite reliability (e.g., Cronbach’s alpha) are not necessarily     scores. For example, beyond the LSA models trained on the TASA
          always done before using scores for analysis, and still rarer are    and EN 100k corpora that have been used previously in creativity

             erties of a set of DT items. Given that such formalized descriptions     researchers and could potentially be better for the quantification of          broadly.      studies of the underlying dimensionality and measurement prop-     research, other text-mining systems are also freely available to
           of the way DT prompts are aggregated into psychometric scores     Originality then LSA. One such system comprises the Googlepublishers.         are rare in the creativity literature, it is difficult to build convincing   News model associated with the Word2Vec algorithm (Mikolov,
                                                                      Chen, Corrado, & Dean, 2013)—which is trained on 100 billionallied        argumentsscoring system,for theincludingreliabilitytheandLSA-basedvalidity ofOriginalityany DT measurescores. Foror               disseminated                                                                words scraped from the website Google News using a more
its        example, after generating LSA semantic distances for every re-   be                                                             modern neural network-based training approach. In addition, the
of
   to      sponse in the dataset, those responses are often averaged, or    Global Vectors for Word Representation (GloVe; Pennington,
one        perhaps summed, within each prompt for every participant (see    not                                                                        Socher, & Manning, 2014) algorithm provides a series of free,
or is      Forthmann, Szardenings, & Holling, 2020 for a close investigation                                                                                    trained systems, including one based on 840 billion words scraped
    and       into the effects of these methodological choices). Then, if multiple    from across the Internet. The GloVe system, by virtue of  its
       DT prompts (or specifically AUT prompts) were administered,
     user       participant scores across those prompts need to be aggregated in a     probabilistic (and therefore possibly more stable) statistical under-Association                                                                           pinnings and massive training corpus, may be more capable of
            reliable way (e.g., CFA) so that a score that represents participant
                                                                          producing reliable and valid Originality scores than previously
            level Originality can be produced. However, the psychometric
                                                                         used text-mining systems, although such a research question has             individual      properties of such a scoring model  (if one is used) are rarely
                                                                           never been systematically addressed. These models are described
    the      reported in the literature, limiting what is known about LSA-based
                                                                                     in more detail in the methodology section of this article, and arePsychological of      Originality scores. Dumas and Dunbar (2014) were an exception to
                                                                                generally distributed as vector spaces providing a mapping of
    use        this, and they found relatively strong evidence for the psychomet-                                                                    words to latent dimensions, which can be used programmatically
             ric reliability of LSA-based Originality scores, but they did not
                                                                                     to measure semantic distance.American        examine the relation between those scores and human-raters or
                                                              As previously reviewed, a number of freely available text-the  personal      Elaboration scores, among other limitations. It should be noted that
                                                                      mining models that have been trained on existing corpora of textby the      the lack of strong psychometric evidence is a problem across much
                                                                     and that are designed to represent the semantic structure of lan-
    for      of the creativity research literature, not just LSA-based Originality
                                                                      guage through the estimation of word vectors within semantic            scores, but the problem may be particularly poignant here, where
                                                                           space exist in the  literature  (i.e., TASA, EN100k, word2vec,        solely      the automated nature of these scores make the large-scale mea-
                                                                        GloVe). As will be delineated further in the Method section of thiscopyrighted        surement of DT possible and provide opportunity for higher-stakes
is                                                                                              article, each of these text-mining models essentially utilizes a           applications of creativity assessment, where low reliability can
                                                                              dimensionality reduction technique to quantify the relations among          intended      pose serious scientific and ethical problems.
   is                                                                words or phrases by examining the angles among vectors (Lan-document                                                                         dauer et  al., 1997; see Figure 1). However, another technique
         article         Moving Beyond LSA in Creativity Research            exists within the psychological literature that offers an alternativeThis
     This       Based simultaneously on the psychometric and psychological     to all of these dimensionality-reduction-based methods: the net-
          evidence in support of text-mining-based Originality scores, as    work modeling perspective (De Deyne, Verheyen, & Storms,
           well as more pragmatic and practical considerations such as the    2016; Kenett, Levi, Anaki, & Faust, 2017). In this body of work,
           speed, cost, and objectivity of these methods, the continued use of     the semantic distance among words is not quantified by examining
           text-mining models to score DT tasks seems justified and desir-     the angles among word vectors, but instead the length of a network
            able. However, because of specific methodological concerns about     path between two words (i.e., the number of intervening words in
       LSA as a method generally, the TASA corpus specifically, as well     the network) is used as a measure of semantic distance. Recent
           as a general lack of formal psychometric investigation into these    work in cognitive psychology (Kumar, Balota, & Steyvers, 2019)
           scoring systems, thinking  critically about other possible  text-    has compared the network science approach to understanding
          mining systems for our work, beyond LSA TASA, appears impor-    semantic  distance and dimension-reduction models LSA and
             tant. Indeed,  it may be that the free availability of the TASA-    word2vec, with some results showing advantageous properties of
            trained LSA tool—as well the fact that  it was the text-mining     the network approach. In the current study, the possibility for
          system originally chosen by Forster and Dunbar (2009)—drives     incorporating network models into a computational psychometric
           the creativity research literature’s choice of this text-mining sys-    approach to divergent thinking (see Kenett, 2019 for an overview)
<a id="page-7"></a>

### PDF 第 7 页

6                                 DUMAS, ORGANISCIAK, AND DOHERTY


              is not investigated, although it is discussed later in this article as a    an object as possible within a certain amount of time (i.e., two
           future direction.                                                 minutes per object in this case). The AUT has been used for
                                                                                assessing divergent thinking and creative ability for decades (Guil-
                     The Current Study                            ford, 1967; Hudson, 1968; Torrance, 1972) and remains one of the
                                                                             most-often utilized tasks within the creativity research literature
           Given the current state of the creativity literature surrounding      (e.g., Dumas & Strickland, 2018; Puryear, Kettler, & Rinn, 2017).
           the quantification of Originality via text-mining models, coupled    The following 10 object names were presented to participants in a
          with the relatively recent availability of text-mining systems sub-    randomized order: book, fork, table, hammer, pants, bottle, brick,
            stantially more advanced than LSA and the TASA corpus, we have      tire, shovel, and shoe. In this investigation, 10 AUT prompts
          undertaken a systematic study of four major freely available text-     (rather than a single AUT prompt as is often the case in creativity
          mining systems (explained in more detail in the Method section)     research) were used to reduce the stimuli dependence of the
         and the  reliability and validity of the Originality scores they     Originality scores (Barbot, 2018). This issue of stimuli depen-
           produce. These systems all rely on a different mix of methods,     dence, and the concomitant need for multiple DT indicators, may
           technical implementations, and training corpora, and we seek to    be even more critically important when scoring the AUT with
          understand which systems are more appropriate for scoring AUT     text-mining systems, because different AUT prompts (e.g., Book)          broadly.
            Originality. Specifically, we aim to assess the internal consistency   may be represented in any given corpus differently, and thereforepublishers.        and factor reliability of AUT Originality scores produced by these     multiple stimuli are needed to produce the most reliable and valid
           text-mining systems at both the scale and latent-variable levels and     scores. In this investigation, the 10 object names that were pre-
allied       compare that reliability to that of human-raters who judged the     sented to participants were chosen to be in line with past work               disseminated
its         Originality of each AUT response. In addition, the predictive     within the DT literature that has incorporated a text-mining ap-   be
of   to       validity of the five Originality scoring systems (human raters and    proach (e.g., Dumas & Dunbar, 2014), as well as common practice
one         four different text-mining systems) is examined in terms of their     within the DT assessment field, where objects are typically chosen    not
or is       correlation to a number of theoretically relevant DT dimensions     that are expected to be familiar to participants, and  that are
    and         (i.e., ideational Fluency and Elaboration), creative personality     reasonably different from one another to provide a reasonably
            characteristics (i.e., Openness and Intellect) and self-reported real-    broad sampling of object types, therefore reducing dependence on
     user      world creative activities. The overarching goal of this investigation    any one stimuli (Acar & Runco, 2019). Scoring procedures andAssociation              is to provide creativity researchers with psychometrically sup-     resulting reliability and validity evidence for AUT scores are the
           ported recommendations as to how to score DT responses for    main focus of this investigation, so that specific information is             individual       Originality,  and  what  general  predictive  patterns  to  other     presented later in the Results section of this article.
    the       creativity-related constructs (e.g., Elaboration) may be expected      Big Five Aspects Scale.  The Big Five Aspects Scale (BFAS;Psychological of      depending on what scoring system researchers choose to use.       DeYoung, Quilty, & Peterson, 2007) is a widely utilized self-
    use                                                                              report personality measure in which participants indicate levels of
                          Method                                   five principal aspects of personality, each of which is dividedAmerican                                                                                   further into two facets. The “big five” dimensions of personality—the  personal                                                                        Neuroticism, Agreeableness, Conscientiousness, Extraversion, and         Participants
by the                                                             Openness—are all available on this measure, but the Openness
    for        This study, which was part of a larger and ongoing investigation     scale is of particular interest to the present investigation, because
        solely       into57.6%)theparticipants.psychometricsParticipantsof  creativity,were recruitedincluded for92this(53studyfemale;via      thiswith dimensiondivergent ofthinking,personalitybothis inthetheorymost perennially(Hornberg &associatedReiter-copyrighted        Amazon Mechanical Turk, a crowdsourcing platform widely used    Palmon, 2017) and in empirical findings (Furnham, Crump, &
is
            in psychology research, including creativity research (e.g., McKay,    Swami, 2009). The Openness dimension is further divided into two          intended         Karwowski, & Kaufman, 2017). Because of the high language    facets—Openness and Intellect—and both of these facets have
   is
         demands of divergent thinking tasks, participants were required to    been shown to be significantly and positively related to divergentdocument
           report themselves as fluent English speakers to participate, al-     thinking and creative outcomes, and are considered the core of the         articleThis        though two participants (2.1%) reported English as their second     creative personality (Oleynick et  al., 2017), making them both
     This      (but fluent) language. Participants were compensated $3.00 each     useful validity criteria in this study. In particular, we conceptualize
            for their participation. Participants were required to be over the age     the Openness and Intellect facets as providing important validity
           of 18 to participate, but the minimum actual participant age was     information in the following way: Intercorrelations among the
           21, with a maximum age of 68. The mean age of participants was     text-mining-based Originality scores and the Openness and Intel-
         37 (SD    10.58). The majority of participants (n    68; 73.91%)      lect facets should, if the validity of the text-mining methods is
           reported  their  race/ethnicity  as European American, whereas     upheld, be similar to the intercorrelations of the Openness and
           smaller proportions of the sample reported their ethnicity as Afri-     Intellect facets and the human-rater-based Originality scores. If
          can American (n    6; 6.5%), Asian (n    9; 9.8%), Latinx (n    5;     Originality scores from one or multiple text-mining models were
           5.43) or multiple ethnicities (n    4; 4.2%).                            to display correlations with Openness and  Intellect that were
                                                                                     substantially different from human-judged Originality, the validity
                                                                                of that text-mining model would be called into question.
        Measures and Tasks
                                                                        Although the most common method used to score self-report
            Alternate uses task.  The AUT is a psychometric measure in    measures like the BFAS in psychology research is through sum-
         which participants are asked to generate as many creative uses for    ming the items, the summation of scores makes a number of strict
<a id="page-8"></a>

### PDF 第 8 页

MEASURING ORIGINALITY                                       7


         measurement assumptions that are unlikely to hold (McNeish &     latent  factor,  latent  factor  reliabilities based on loadings and
          Wolf, 2020). Therefore the Openness and Intellect scores for this     uniquenesses of H    .952 and       .934. For future analysis,
           study were generated through confirmatory factor analysis (CFA)     empirical Bayes-based latent factor scores were computed for each
         by fitting a two-factor correlated model to both scales at once, as     of the six administered ICAA scales, as well as a total Creative
        DeYoung and colleagues (2007) intended and validated. Specific     Activity that incorporated all six scales.
          psychometric information for each scale appears below.                In this study, the ICAA is included as a validity-criterion mea-
            Openness.  The 10-item Openness facet of the BFAS has been     sure with which to correlate the Originality scores produced by
            particularly associated with creative outcomes, because it features    both human raters and the various text-mining models. In general,
          such self-report items as I need a creative outlet and I believe in   we conceptualize this validity procedure as requiring the text-
           the importance of art. Participants indicated the degree to which    mining-based Originality scores to approximate, in their correla-
          each statement was true of them by dragging a 100-point slider     tions to the ICAA, the nature of the human-rated Originality
          with poles of 100    strongly agree and 0    strongly disagree.     scores. This validity-criteria procedure is based on the general
           After reverse-coding all negatively worded items, in this study, the    problem in creativity research that an ongoing need to utilize
         10 items on the Openness facet of the BFAS achieved a scale    human raters in our work creates a bottleneck to scaling creativity
            internal consistency of      .839, with latent factor internal con-     research to very large data sets. Here, we seek to test the capability          broadly.
           sistency indices based on factor loadings and uniquenesses being     of the text-mining models to create Originality scores for the AUTpublishers.     H   .896 and      .861. Openness scores were generated from the     that are similar to those produced by humans, but much more
allied      CFAIntellect.model viaTheempiricalintellectBayesfacet andof thesavedBFASin thealsodataset.contained 10     rapidlyexternalandvalidityat a muchcriterionlowerto cost.ascertainHence,whetherthe ICAAthe servestext-miningas an               disseminated
its          self-report items, including I like to solve complex problems and I    models are successful at accomplishing this goal.   be
of   to    am quick to understand things. Participants responded to these
one        items in the same manner (i.e., with a slider) as they did the items    not                                                            Administration Procedures
or is     on the Openness facet. After reverse-coding all negatively worded
    and      items the 10 items on the Intellect facet of the BFAS achieved a       All participation for this study was conducted online via Me-
           scale reliability of     .840, with latent factor internal consistency     chanical Turk, and the study website itself (which participants
     user      indices based on factor loadings and uniquenesses being H   .876    were provided a link to) was hosted by Qualtrics. Informed consentAssociation
         and     .858. Intellect scores were generated from the CFA model    was obtained before participants could move forward with the
           via empirical Bayes and saved in the dataset.                      measures (these procedures were approved by the institutional             individual        Inventory of Creative Activities and Achievements (ICAA).    review board at the Institution where the study took place). Study
    the     The ICAA is a relatively recently developed (i.e., Diedrich et al.,     instructions asked participants to complete the measures withPsychological of      2018) self-report measure for real-life creative activities and ac-    minimal distractions and recommended that they turn off elec-
    use      complishments across eight domains: literature, music, arts and     tronic devices as well as close other websites or programs open on
             crafts, cooking, sports, visual arts, performing arts, and science and     their computer. Because the AUT requires a significant amount ofAmerican         engineering. Given the general nature of this sample, and time-     typing, participation required a traditional keyboard and participa-the  personal       constraints on the data collection, we administered the creative     tion via smartphone or tablet was not allowed. Participants were
by the       activity scale (rather than achievement) for six of those original    given two minutes to provide uses for each object before they were
    for      eight domains: music, literature, arts and crafts, cooking, visual     automatically advanced to the next object, and they could not
        solely        arts,Likert-styleand performingitems that arts.ask participantsEach of thesehowscalesmany consistedtimes theyofhavesix    advance10 objectsbefore(i.e.,thoseaftertwo20 minutesmin), participantswere up. Afterwererespondinginformedtothatallcopyrighted        done particular creative activities in the past 10 years with five     the task was complete, and moved to the self-report portion of the
is        response categories: never, 1–2 times, 3–5 times, 6–10 times, and     study. In this phase, participants first provided responses to the          intended     more than 10 times.                                  ICAA and then moved to the BFAS. Finally, participants re-
   is        For example, in the music domain, participants are asked how    sponded to the demographic question and logged out of the studydocument        many times they have written a piece of music, or created a mix     website.         articleThis          tape, among other items. In the arts and crafts domain, participants
     This      are asked how many times they created an original decoration. In                                           AUT Scoring Procedures
           cooking, how many times they made up a new recipe. The visual
         and performing  arts scale asks how many times  participants      The main focus of this investigation was to examine the reli-
           painted a picture and performed in a play, respectively. In this     ability and criterion validity of multiple Originality measurement
          sample, each of the scales of the ICAA achieved satisfactory scale    methodologies for the AUT. As such, the AUT was scored a
             reliability as well as satisfactory latent factor reliability based on    number of different ways in this study, each of which is detailed
            scale-specific single-factor CFA models, with literary activities    below.
          having the lowest reliability (      .800; H    .837;         . 814),       Fluency.  As a criterion by which to examine the validity of
          music activity having the highest (     .909; H   .968;      .915),     Originality scoring methods, the AUT was scored for Fluency.
         and the other scales (visual arts;      .826; H    .908;      .836;      First, the number of uses generated by each participant for each
          cooking;      .874; H     . 903;      .876; performing;      .876;     object was tallied, and then summed across all 10 items on the
      H     . 899;        . 882; crafts;      .90; H     . 921;        . 905) being   AUT, producing a “total-uses” variable for analysis. Counts such
            in the middle. Taken together, all 24 items on these six scales     as these are the principal way in which fluency has been opera-
           displayed a composite scale reliability of     .926. and, as a single     tionalized in the extant literature (Plucker & Makel, 2010). In this
<a id="page-9"></a>

### PDF 第 9 页

8                                 DUMAS, ORGANISCIAK, AND DOHERTY


            investigation, fluency counts across the 10 items on the AUT    number of responses being completely unoriginal (i.e., 0) or very
           exhibited a high level of scale internal consistency (      .946).    high on the Originality scale (i.e., 4). Such a continuous distribu-
         However, to avoid making potentially untenable measurement     tion of Originality—rather than discrete Originality categories—
           assumptions, Fluency scores were generated via empirical Bayes    has been empirically documented in the literature (e.g., Dumas,
         from a single factor CFA model fit the 10 Fluency indicators. The     2018).
           scale exhibited latent factor reliability indices of H    .962 and      The four human coders coded the 5,491 responses with a “fair”
                 .957.                                                               level of interrater agreement (Fleiss’     0.2198; Fleiss & Cohen,
            Elaboration.  Also following well-established scoring proce-     1973). Typically, within the psychological research literature, any
           dures in the divergent thinking literature (e.g., Forster & Dunbar,     lack of exact agreement among coders would be resolved through
          2009; Torrance, 1988), participant Elaboration scores were calcu-     discussion until all coders were able to converge on an agreed-
            lated by averaging the number of words utilized per response    upon categorical rating for every response (Gwet, 2014). Such a
           within each of the AUT prompts. In this scoring procedure, aver-    method operates under the measurement assumption that there is a
          aging within the AUT prompt is meant to reduce the implicit     true Originality category for each generated response  (i.e., the
           association between Elaboration and Fluency (i.e., those partici-     latent Originality attribute is ordinal), and therefore coders must
           pants who generated more responses will have used more words in    work to sort the generated responses into their true categories.          broadly.
              total, but perhaps not on average). However, a statistical relation    However, an alternative method would assume that the 0–4 Orig-publishers.        between these two dimensions of divergent thinking may still exist     inality categories the coders used were underlain by a continuous
allied         regardlessical associationof this betweenscoring choiceideationalbecausefluencyof a possibleand the psycholog-ability to     latentcategoriesdistribution,are meantandto indicatethereforelocationsthe originallyon thatcodedcontinuousOriginalitylatent               disseminated
its         elaborate on those ideas (Hudson, 1968). In addition, the strength     distribution.   be
of   to      of the relation between Elaboration and Originality has been the     Common in crowdsourcing methodology (e.g., Organisciak, Te-
one        focus of previous investigations of text-mining scoring systems for     evan, Dumais, Miller, & Kalai, 2014; Snow, O’Connor, Jurafsky,    not
or is      Originality (Forthmann et al., 2019), and therefore it is of high   & Ng, 2008), where varying judgments of quality from raters are
    and      importance here. These 10 AUT prompt elaboration scores dis-     regularly aggregated, this continuity assumption suggests that ex-
          played a high level of scale reliability (     .958), and at the latent     act categorical agreement among raters is not crucial, because,
     user       factor level (via a single factor CFA), displayed strong latent    over multiple raters, a consensus about where on the underlyingAssociation
            internal consistency indices H   .965 and      .961. Elaboration     Originality distribution each generated response may be located
           scores for each participant were generated via empirical Bayes    can arise through averaging the ordinal category codes across             individual     from the single factor CFA model.                                         raters. The intuition here is that disagreement among raters on
    the         Originality.   Originality in this investigation was scored using     ordinal codes is actually instructive and valuable to researchers.Psychological of     two main categories of methods: human raters and text-mining    For example, if three raters coded a particular AUT response as a
    use      models. In addition, the reliability and validity of scores produced     ‘3’ on the Originality scale, but one rater coded it as a ‘4,’ that last
         by a number of different types of text-mining models are com-     rating is still considered a nudge toward the more novel end of theAmerican         pared.  It should be noted here that, given critically important     scale for that response, and an averaged Originality score for thatthe  personal      concerns about the way that any text-mining system deals with     response of 3.25 would be considered closer to “true” than simply
by the    common function words (Forthmann et al., 2019), all of the anal-     the modal rating of 3. For this reason, we created an aggregated
    for       ysis in this study utilizes inverse-document-frequency (IDF) term-    human-coded Originality rating for each AUT response by aver-
        solely      weightingextremely common(Robertsonwords& Jones,(e.g., “is”).1976)Despitecorrectionsdifferencesto dealin withhow    agingresponses.each Then,of the to4 coders’aggregateratingsthoseforresponse-levelevery one ofOriginalitythe 5,491copyrighted         they were developed, each scoring system provides a model of     ratings to the participant-level, we further averaged each of those
is        language in a linear space, aiming for comparable distances be-     response ratings within each AUT prompt (e.g., Book) for every          intended     tween words in English. To score from each system’s model, a     participant. This procedure resulted in 10 prompt-level human-
   is      weighted sum of word vectors is taken to represent each phrase for     rated Originality scores for each of the 92 participants in thedocument          a response, and the cosine distance is taken between the response     dataset. Because it is the main focus of this investigation, further         articleThis        and AUT prompt (see Figure 1).                                      analysis with these human-rated Originality scores (e.g., modeling
     This      Human raters.   First, every generated response from the 92    an underlying latent Originality attribute across all AUT prompts)
           study participants across the 10 items on the AUT was coded for     are included in the Results section of this article. Further, the issue
           Originality by four human coders. In all, 5,491 responses were     of the interrater reliability of these human-rated Originality scores
           generated to the AUT in this study, with an average of 55.81      is returned to with a critical lens in the Discussion section.
         (SD   31.72) per participant. The first Originality coder was the      Text-mining systems.  Here, we systematically compare the
            third author of this article, and the other three were paid research     capability of four different publicly available text-mining systems
            assistants. Each coder was instructed to score each generated     to create reliable and valid participant Originality scores on the
          response from 0–4, with zero being “totally ordinary” and four   AUT. Each of these four text-mining systems differ in a variety of
          being “maximally novel.” Coders were specifically trained to    ways, including the corpora of text that they are trained on, the
           conceptualize most responses as being likely to fall toward the     parameterization and specification of the statistical models they
          middle of that 5-point Originality scale: a belief that reflects our     use, and the way they correct for difficult-to-model aspects of
          assumption that the Originality of generated responses is based on     real-world language use such as words with multiple meanings and
          a continuous distribution, manifesting such that most responses are    synonyms. Generally, larger corpora will more accurately repre-
            in the middle of the Originality scale  (i.e., 2) with a smaller     sent the relations between words in the language, though the
<a id="page-10"></a>

### PDF 第 10 页

MEASURING ORIGINALITY                                       9


         domain of the documents will lead to differences in how the    has been outstripped by other systems in terms of corpus size and
          language is interpreted and may affect the transferability of that    model sophistication (Crossley et al., 2017). For example, recent
            particular model. For example, is bank more associated with river    work in the information sciences has confirmed Landauer and
           or money? A naive algorithm learning English from a collection of    Dumais’ (1997) argument in showing that LSA spaces trained on
         documents will decide that answer differently based on what those   TASA do tend to match human semantic judgments but has also
         documents were written about. The sizes and domains of the    found better performance with larger corpora (S¸tefa˘nescu, Ban-
          corpora on which each system was trained are noted in Table 1.     jade, & Rus, 2014). In this study, we use the publicly available
           Different systems may also correct for perceived importance of   TASA model trained by Günther et al. (2015).
          words, deemphasizing common function words (e.g., and, the) or       English 100k LSA.  Also originally trained by Günther, Dud-
          removing them altogether. Finally,  all the system models are     schig, and Kaup, the English (EN) 100k LSA text-mining system
            trained using different training methods. These methods differ on    was previously applied to divergent thinking task data by Forth-
           choices such as what frame of document suggests a relationship    mann and colleagues (2019). This system is trained on a concat-
          between words and how the training algorithm implements that     enation of multiple general purpose corpora of texts: a Wikipedia
           theory. The choice of how many latent dimensions are learned also    image, the general text British National Corpus, and a web crawl
            affects the system: Too few dimensions will lack depth and dis-    corpus that together included more than 5 million documents.          broadly.
           criminatory value between words, whereas too many will overfit to     After an  initial modeling of the language in these 5 millionpublishers.         the documents. In the current investigation, we do not attempt to    documents, the 100,000 most frequently occurring unique words
allied         absolutelytraining ofcontrola text-miningfor everymodel.possibleInstead,methodologicalwe focusoptionon alreadyin the    werealthoughretainedthe LSAto buildtrainingthe systemmethod(hencein thisthesystem100kisinthethesamename).as thatSo,               disseminated
its         created text-mining systems that creativity researcher are currently     in the TASA system, the size and generality of the corpora used in   be
of   to      able to access free of charge, to provide a meaningful demonstra-      this the EN 100k system may be more advantageous for the
one         tion of the strengths and weaknesses of each extant system spe-     quantification of originality on DT tasks because the more general    not
or is       cifically in the context of creativity and divergent thinking re-     corpora on which this model is trained may better represent the
    and      search.                                                                   true semantic space from which DT task participants draw their
           Each of the four text-mining systems that are tested in this study     responses.
     user      are succinctly explained below. A bulleted explanation of each of      Global Vectors for Word Representation 840B.  Publicly avail-Association
           these text-mining systems also appears in Table 1. Analysis based     able through the Stanford natural-language-processing laboratory
         on these text-mining systems was accomplished by remotely ac-    (Pennington et al., 2014), but never before applied to the analysis             individual      cessing their freely available systems via the Python programming     of divergent thinking task data, the Global Vectors for Word
    the      language. All reproducible computational code used in this inves-     Representation (GloVe) 840B text-mining system was trained on aPsychological of       tigation are freely available online via our laboratory ongoing    corpus of 840 billion words that were scraped from a variety of
    use      Github account (https://github.com/massivetexts), and a static de-     online sources including Wikipedia and Twitter. Although GloVe
           pository of the code used for this study is also available on the      is similar to the LSA-based text-mining systems in that its goal isAmerican       Open Science Foundation (https://github.com/massivetexts). In ad-     to quantify the semantic relation between two words or phrasesthe  personal       dition, computational code  is available as online supplemental     within a geometric space, GloVe accomplishes this goal through a
by the      materials published with this article.                                    probabilistic modeling framework. In addition, GloVe calculates
    for        Touchstone Applied Science Associates (TASA) LSA.  This sys-     correlations among terms in a more targeted way than does LSA:
        solely      tem,literaturewhichonisdivergentby far thethinkingmost commonly(e.g., Dumasapplied& Dunbar,in the extant2014;    bytermexaminingwhere  ita issmallused,windowrather ofthanwordexaminingco-occurrenceco-occurrencearound eachincopyrighted         Forster & Dunbar, 2009; Forthmann et al., 2019), is trained on a      full-text documents. This shift in the mathematical and statistical
is        corpus of 37,651 educational texts and was originally intended to     underpinnings of the text-mining systems may hold potentially          intended     mimic to expected reading experience of the typical entering     positive impact on the measurement of AUT originality in that it
   is      undergraduate student. This system was used by Landauer and   may potentially produce more stable and reliable estimates ofdocument         Dumais (1997) in their initial demonstration of the capability of     response Originality (and this hypothesis will be tested in the         articleThis      LSA to approximate the human semantic relations, but since then     current study).
     This

          Table 1
           Short Description of Text-Mining Systems Included in This Investigation

           System name                 Training corpora                               Training scale                              Reference

        TASA LSA      Multi-subject educational texts                37.7 thousand documents (92.4 thousand       Landauer & Dumais, 1997
                                                                        unique words)
        EN 100k LSA   Wikipedia, ukWaC (web crawl), and British   5.4 million documents (2 billion words, 100    Günther, Dudschig, & Kaup, 2015
                               National Corpus (general)                   thousand unique words)
          GloVe 840B   Common Crawl (web documents from sites   840 billion words (2.2 million unique words)   Pennington, Socher, & Manning, 2014
                                including Wikipedia and Twitter)
         Word2Vec      Google News (articles)                    100 billion words (3 million unique words)    Mikolov, Sutskever, et al., 2013

            Note.  TASA    Touchstone Applied Science Associates; LSA    Latent Semantic Analysis; EN    English; GloVe    Global Vectors for Word
            Representation; Word2Vec    word-to-vector.
<a id="page-11"></a>

### PDF 第 11 页

10                                DUMAS, ORGANISCIAK, AND DOHERTY


           Word2Vec.  Named for the “word-to-vector” methodology  it    Table 2
          employs, Word2Vec focuses specifically on word-level corpora    Composite and Factor Reliability Indices for Originality Scores
          scraped from massive online sources of text (Mikolov, Chen, et al.,    From Each Scoring System
           2013). This text-mining system was created at, and is publicly
           available through, the tech company Google, and was trained on a     Scoring system                               H
          corpus of 100 billion words scraped from the news-aggregator    Human raters                .943              .948              .952
         Google News. Word2Vec modeling methodology focuses on the   TASA LSA                 .813              .825              .867
           context of individual words, and through a neural network predic-   EN 100k LSA               .730              .758              .825
            tive modeling approach, works to predict a target word from a    GloVe 840B                 .800              .807              .875
                                                                    Word2Vec                  .741              .743              .807
          sample of closely co-occurring words  (i.e., context words). In
        Word2Vec parlance, this method is termed “skip-gram,” because     Note.  TASA   Touchstone Applied Science Associates; LSA   Latent
           the model skips individual target-words when training and then     Semantic Analysis; EN    English; GloVe    Global Vectors for Word
                                                                                         Representation; Word2Vec    word-to-vector.
            predicts the skipped word based on the context-words that co-
          occur with  it. Previous research has found that this Wor2Vec
         method preserves the true semantic relations among words more          broadly.
            effectively than other training models such as LSA (Mikolov,      Confirmatory  factor  analysis. A  unidimensional CFApublishers.         Sutskever, Chen, Corrado, & Dean, 2013). Further, Word2Vec    model, in which all 10 AUT prompts loaded on a latent Originality
allied        modelsamong wordsperform(Bianchiwell at &identifyingPalmonari,high-order2017). Becauseanalogicalof relationsthe cog-     factor,inality scoringwas fit tosystemsitem-scores(pleasegeneratedsee Figureby2eachfor aofconceptualthe five Orig-path               disseminated
its          nitive similarity between analogical and divergent thinking (e.g.,    diagram of this CFA model). Theoretically, such a model corre-   be
of   to     Green et  al., 2012), this finding may suggest that Word2Vec    sponds to a measurement assumption that all the administered
one        methods are particularly suited for the measurement of originality   AUT prompts (when scored for Originality) indicate a single    not
or is       in DT tasks.                                                       underlying originality construct and therefore represents common
    and                                                                measurement practice in the creativity literature (e.g., Storme et
                                 Results                                             al., 2017). These CFA models were fit using maximum likelihood
     user                                                                           estimation in Mplus Version 8.0 (Muthén & Muthén, 2019). BasedAssociation         The analysis for this psychometric investigation of text-mining-
                                                                  on the model root mean square error of approximation (RMSEA;
          model-based Originality scoring systems unfolded in the following
                                                                     See Table 3 for exact values), none of these unidimensional
            stages: (a) a careful investigation of the reliability of participant             individual                                                                 methods achieved a level of model-data fit that would be consid-
           Originality scores generated by both human raters and text-mining
    the                                                                         ered ideal in the methodological literature (i.e., below .06; Hu &
           systems, with an eye toward both composite and latent factorPsychological of                                                                              Bentler, 1999; McNeish, An, & Hancock, 2018). However, the
             reliability; (b) an analysis of the correlations among human rated
    use                                                                 models for both the human raters and the GloVe text-mining         and text-mining system generated Originality scores; and (c) a
                                                                        system achieved a level of fit that would meet current standards in
            criterion validity analysis of Originality scores in which the cor-American                                                                               the creativity literature, where measurement model-data fit is often
            relations from both human rated and text-mining system generatedthe  personal                                                                                   slightly weaker than in more traditional measurement areas such as           Originality to Fluency, Elaboration, Openness, Intellect, and Cre-
by the                                                                         reading or math (e.g., Yoon, 2017).
            ative Activities were examined. Each of these three analytic stages
    for                                                                             In addition, although the scoring systems differed in the strength
           are explained, and results are presented, below.
        solely                                                                           ofvidualtheirAUTCFAitemsloadingsalso (seedisplayedTable general3 for loadingtrends details),in the strengththe indi-of          Reliabilitycopyrighted                                                                                        their loadings across scoring systems. For example, the prompt
is           Here, reliability of each of the five included Originality scoring    Rope displayed weaker standardized loadings than other prompts          intended     methods (i.e., human raters, TASA LSA, EN 100k LSA, GloVe     across multiple of the scoring systems, whereas the prompt Bottle
   is     840B, and Word2Vec) is examined using both observed variable     displayed stronger loadings across multiple scoring systems. Thisdocument               (i.e., Classical Test Theory [CTT]) and latent variable (i.e., Con-     pattern may likely be attributable to differential participant famil-         articleThis         firmatory Factor Analysis [CFA]) methods.                               iarity with certain objects, or perhaps the actual functional capa-
     This       Composite internal consistency.  Human-coded Originality      bilities of each object to facilitate original alternate uses. One
            ratings on the 10 AUT prompts displayed a high level of composite    anomaly in these general patterns were the extremely weak stan-
           or scale reliability (see Table 2 for reliability coefficients). In     dardized loadings for Book and Table in the EN 100k LSA system,
            contrast, the composite reliability of the text-mining system gen-    and the model-data  fit or  this scoring system was also poor
           erated Originality was substantially lower, although the TASA    compared with the other models, so that may indicate that this
       LSA and GloVe 840B each reached levels of composite reliability    corpus does not well-represent the true semantic relations among
            that may be generally considered acceptable in the psychological    Book, Table, and their associated uses.
           research literature (i.e., .80 or above). These coefficients are es-      Many-faceted Rasch analysis for human raters.  The con-
            pecially important given the propensity of creativity researchers     firmatory factor modeling perspective above intentionally aggre-
            for using summed-scores, rather than optimally weighted latent     gated human rated originality across all the responses to a given
           variable scores, in their research. However, a more in-depth anal-   AUT prompt (e.g., Book), for all four of the human raters, through
            ysis of the measurement properties of the Originality scoring     averaging. This method was designed to treat the raters as equally
          systems is necessary to understand the way these scores relate to a    weighted voters in terms of the originality of a given AUT re-
            latent Originality construct.                                         sponse, and therefore also for participant originality scores. How-
<a id="page-12"></a>

### PDF 第 12 页

MEASURING ORIGINALITY                                      11





          broadly.
publishers.
                              Figure 2.  Conceptual path diagram of the latent measurement model used to determine the reliability of the
allied                               Alternate Uses Task (AUT) scoring methods.               disseminated
its
   be
of
   to       ever, a reasonable alternative perspective, recently demonstrated    each prompt as was used for the CFA above). MFRM parameter
one    not     by Primi and colleagues (Primi, Silvia, Jauk, & Benedek, 2019)     estimates are presented here in Table 4, which contain the diffi-
or is
         would be to employ a Many-Facet Rasch Model (MFRM) to     culty/severity  for the AUT items and  raters, as well as the
    and      incorporate differences in the leniency or severity of individual     parameter-theta (latent score) correlations. As can be seen in this
     user     human judges  into the  calculation of  originality  scores. The     table, some AUT items were more difficult for participants to thinkAssociation     MFRM model has been previously demonstrated to be useful in     of original uses for (e.g., Shoe; difficulty    .60), whereas other
            creativity research, in particular in modeling measurement error     items were easier (e.g., Brick; difficulty      .53). Similarly, some
           associated with multiple human raters (Barbot, Tan, Randi, Santa-    human raters were more lenient in judging originality of responses             individual
          Donato, & Grigorenko, 2012), and this previous usage suggests      (e.g., Rater 3; severity    1.4), whereas others were more severe
    the           relevance of this modeling tool to the current study. So, as a further      (e.g., Rater 2      .66). As a general measure of internal consis-Psychological of
           point of comparison to the CFA approach presented above, an     tency, the Rasch average reliability among the 10 AUT items in the
    use      MFRM with three facets (i.e., judges, items, and participants) was   MFRM was .90, and therefore MFRM based scores were gener-
               fit to these data using the specialized computer program Facets     ated and saved in our dataset for future analysis. As an alternativeAmerican
           (Linacre & Wright, 1988), and used to calculate originality scores     to MFRM not applied here, interested readers should also see thethe  personal            for each participant in the dataset. It should be noted that the use     application of item-response models to data from multiple human-
by the           of the term “facets” is different in this context than in the person-     raters recently posited by Myszkowski and Storme (2019).
    for             ality measurement context. In the personality measures used in this      Latent factor internal consistency.  Using the standardized
           study, the facets refer to the finely grained subscales within the Big     loadings and residual variances that are generated when fitting the        solely         5 factors. In MFRM, a facet refers to a source of measurement   CFA models, we then calculated two modern factor reliabilitycopyrighted
is          error, in this case error may arise from inconsistencies among the      statistics for each of the scoring systems: Omega (McDonald,
        human raters, AUT items, or the distribution of participant original    1999) and H (Hancock, 2001). Although both of these indices          intended
   is      thinking ability.                                                       represent a more sophisticated estimate of the score reliability thandocument        To accommodate the MFRM here, modal ratings for each judge    Cronbach’s alpha, they differ in their assumptions about the way
         article     were utilized for each AUT prompt (as opposed to means within     participant scores will be produced in future investigations using aThis
     This
          Table 3
          Confirmatory Factor Model Parameters for Each Scoring System

                                                                                     Alternate uses task prompt standardized loadings

            Scoring system     Model RMSEA     Book      Bottle      Brick     Fork      Pants     Rope     Shoe     Shovel      Table      Tire

         Human raters            .063            .837       .867       .739      .798      .864       .642      .788       .807        .825      .838
        TASA LSA              .082            .732       .807       .655      .429      .414       .433      .744       .327        .497      .562
        EN 100k LSA            .124            .023       .647       .513      .433      .650       .629      .742       .499        .039      .564
          GloVe 840B             .069            .672       .784       .805      .595      .590       .161      .752       .268        .309      .350
          Word2Vec               .079            .605       .761       .488      .100      .469       .336      .686       .457        .421      .334

            Note.  TASA    Touchstone Applied Science Associates; LSA    Latent Semantic Analysis; EN    English; GloVe    Global Vectors for Word
            Representation; Word2Vec   word-to-vector; RMSEA   root mean square error of approximation. All standardized loadings significant at p   .05 except
         Book and Table in the EN 100k LSA system.
<a id="page-13"></a>

### PDF 第 13 页

12                                DUMAS, ORGANISCIAK, AND DOHERTY


          Table 4                                                                            (e.g., a sum) as opposed to using a latent scoring model. Going
        Many Facet Rasch Model Parameters for Human Rated             forward, Originality scores for all 92 participants from each of the
       AUT Items                                                              five scoring system were generated via empirical Bayes using the
                                                 SAVEDATA command in Mplus.
             Model                                           Parameter-Theta
             parameter            Difficulty/Severity (SE)             correlations
                                                                  Relation Between Text-Mining Models and
        AUT Items                                  Human Raters
           Book                        .13 (.09)                       .71
               Bottle                       .12 (.09)                       .78            Perhaps the most critical test of the efficacy of the text-mining
              Brick                        .53 (.10)                       .70
                                                                               scoring systems  is their ability to approximate the Originality
             Fork                         .06 (.09)                       .71
              Pants                        .15 (.09)                       .81           scores produced by human raters. Table 5 holds the correlations
            Rope                        .05 (.09)                       .62        among the five scoring systems included in this investigation. As
            Shoe                        .60 (.09)                       .74          expected, the human rated Originality scores generated via CFA
             Shovel                      .35 (.10)                       .71         and MFRM were correlated very strongly (.98). More critical are
             Table                        .11 (.09)                       .76
                                                                                  the correlations among the text-mining systems and the human          broadly.         Tire                         .09 (.09)                       .76
         Human raters                                                                   raters. These correlations show that the GloVe text-mining system
              Rater 1                      .63 (.06)                       .70             is the most capable of producing Originality scores that resemblepublishers.
              Rater 2                      .66 (.06)                       .69          those of humans. In contrast, the Word2Vec system’s Originality
              Rater 3                      1.4 (.06)                       .71
                                                                                scores were the most weakly correlated with human-rated scores.allied               disseminated         Rater 4                      .11 (.06)                       .70
its                                                                        Although, it should be noted, all four of the systems utilized here
of   be       Note. AUT    Alternate Uses Task; SE    standard error.               produced  latent Originality scores  that were significantly and
   to
one                                                                                  positively correlated with scores from human raters. In addition,    not
or is                                                                             the Originality scores from each of the four text-mining systems
          measure. If, in the future, Originality scores are created by sum-     correlated strongly (in the .9’s) with one another, indicating that—
    and     ming or averaging across multiple AUT prompts, then Omega is     although the systems differed in their observed and latent reliabil-
     user      the best representation of the reliability of those scores, but H      ity indices and CFA model-data fit—overlapping informationAssociation        assumes an optimally weighted measurement model in which the    about participant Originality is provided by each system.
           Originality scores are estimated directly from a CFA or item-
          response model (McNeish, 2018). For this reason, Omega is al-             individual                                                               Criterion Validity
         ways slightly lower than H to account for the added measurement
    the
            error associated with summing or averaging scores across a mea-       Here, the capacity of the five Originality scoring systems toPsychological of
           sure rather than using a psychometric model.                      produce scores that predict other common indicators of creativity
    use          As can be seen in Table 2, the human-raters achieved by far the       (i.e., Fluency, Elaboration, Openness, Intellect, and Creative Ac-
         most reliable Originality scores. However, all of the text-mining      tivities) are systematically examined. See Table 6 for correlationsAmerican
           systems, at least in terms of coefficient H, achieved an acceptable     discussed in this section.the  personal
            level of factor reliability (i.e., above .80) as well, although the most       Fluency.  The theoretical relation between Fluency and Orig-
by the
            reliable system (i.e., GloVe) was substantially more so than the     inality is currently debated in the creativity research literature, with
    for
            least (i.e., Word2Vec). This finding implies that, should research-    some scholars arguing for a positive, zero, or negative correlation
            ers generate scores from a latent measurement model, all of the    among these dimensions of divergent thinking (see Dumas &        solelycopyrighted         scoring systems included here are capable of producing generally    Dunbar, 2014 or Forthmann et al., 2019 for discussions of this
is         acceptable scores (although GloVe scores would be the most     issue). Here, human-rated Originality scores (calculated via CFA)
          intended       reliable, and have the best model-data-fit). In terms of Omega—     correlated weakly and positively (and nonsignificantly) with Flu-
   is     which converged on the same results as Cronbach’s alpha—only    ency scores, implying that that—should we take the human-rateddocument         the TASA LSA system and GloVe achieved acceptable reliability,     scores as baseline truth—the actual relation between these dimen-
         article      indicating that these are the only systems that produce stable     sions is close to zero, at least in this general-population sample.This
     This     enough scores to warrant calculating a simple composite score    However, all of the text-mining systems in this study displayed


          Table 5
           Correlation Matrix Among Originality Scores From All Scoring Systems Included in This Investigation

               Scoring system       Human raters (CFA)    Human raters (MFRM)    TASA LSA    EN 100k LSA     GloVe 840B    Word2Vec

         Human raters (CFA)                1.00
         Human raters (MFRM)               .98                      1.00
        TASA LSA                          .67                        .56                  1.00
        EN 100k LSA                        .66                        .55                    .93              1.00
          GloVe 840B                         .73                        .63                    .97                .96              1.00
         Word2Vec                           .58                        .45                    .96                .95                .94            1.00

            Note.  TASA    Touchstone Applied Science Associates; LSA    Latent Semantic Analysis; EN    English; GloVe    Global Vectors for Word
             Representation; Word2Vec   word-to-vector; CFA   confirmatory factor analysis; MFRM   Many-Facet Rasch Model. All correlations significant at p    .01.
<a id="page-14"></a>

### PDF 第 14 页

MEASURING ORIGINALITY                                      13


          Table 6
           Correlations Among Originality Scoring Systems and External Criteria

               Scoring system               Ideational fluency          Elaboration         Openness            Intellect          Creative activities composite

         Human raters (CFA)                  .176                    .215                 .088               .086                       .093
         Human raters (MFRM)               .103                    .186                 .039               .117                       .100
        TASA LSA                          .346                    .239                 .119               .010                       .077
        EN 100k LSA                        .264                    .273                 .046               .044                       .083
          GloVe 840B                         .336                    .233                 .084               .018                       .077
         Word2Vec                           .365                    .339                 .084               .014                       .025

            Note.  CFA   confirmatory factor analysis; MFRM   Many-Facet Rasch Model.
            p    .05.    p    .01.


            significant and positive correlations (although only moderate in    given that the human rated scores were also positively associated          broadly.      strength) with Fluency. Given this finding, it appears that human-    with Elaboration, it appears that the text-mining systems are not
           rated Originality scores have the greatest degree of discriminant    much more confounded with Elaboration than are human raters,publishers.          validity from Fluency scores, whereas text-mining models under     although substantial variation among the text-mining systems was
allied         investigation produced Originality scores that were much more     observed. Specifically, the GloVe system was most capable of               disseminated      strongly associated with Fluency. In particular, the EN 100k sys-     preserving the low-moderate correlation with Elaboration, fol-
its   be     tem displayed the lowest correlation to Fluency among the text-    lowed by TASA. The Word2Vec system produced the strongest
of
   to     mining systems, making it the most consistent with human-rated     correlation with Elaboration.
one    not      Originality scores in that regard.                                Openness and intellect.  In this study, none of the Originality
or is        Elaboration.  In previous work with text-mining system Orig-     scoring systems (human-rated or text-mining) produced scores that
    and       inality scores (Forthmann et al., 2019), the relation between Elab-     significantly correlated with Openness or Intellect. However, in
     user      oration and Originality has been considered a source of bias in the     the case of both of these creative-personality indicators, the GloVe
            scores. In this investigation, following previous methodological    system produced correlations that were closest to those of theAssociation
          recommendations, the IDF correction for stop-words was utilized.    human raters, indicating that the GloVe originality scores approx-
          Here, we found that the human raters’ Originality scores (calcu-    imated human-rated Originality scores the best in regards to their             individual       lated via CFA) were significantly and positively associated with     relation to creative personality variables.
    the      Elaboration (which is a stronger relation than those human-rated       Creative activities.  When predicting the composite of thePsychological of           scores had with Fluency). In contrast, the correlation between     Creative Activities measure, none of the Originality scoring sys-
    use   MFRM calculated Originality ratings and elaboration was not    tems produced significant correlations, although again the GloVe
            significant (i.e., p    .072). Following the pattern set by the CFA    system was most in-step with the human-raters. At the moreAmerican        produced Originality ratings, all of the text-mining systems also     fine-grained level of the individual scales of the Creative Activitiesthe  personal      produced scores that were significantly and positively associated    measure (see Table 7), the human-rated Originality scores (calcu-
by the      with Elaboration, although all of the text-mining systems displayed     lated either by CFA or by MFRM) did significantly but negatively
    for       correlations to Elaboration that were stronger than that of the     correlate with creative Cooking activities. These findings imply
        solely      human-raters:criminant validitya findingsissues, thatevenhighlightswith the IDFpreviouslycorrection.observedHowever,dis-      that,scoresatareleastnotinrelatedthis general-populationto the self-reportedsample,domain-specificAUT Originalitycreativecopyrighted
is
          intended      Table 7
   is      Correlations Among Creativity Indicators and Creative Activitiesdocument
         article          Creativity indicator              Performance              Arts            Cooking              Crafts              Literary          MusicThis
     This       Originality scoring systems
          Human raters (CFA)                   .120                 .081               .227               .075              .108              .029
          Human raters (MFRM)                 .119                 .097               .243               .081              .093              .044
         TASA LSA                           .156                 .022               .139               .022              .108              .025
         EN 100k LSA                         .155                 .041               .103               .051              .085              .033
            GloVe 840B                          .134                 .038               .152               .063              .107              .053
           Word2Vec                            .116                 .016               .085               .042              .134              .033
            Creative personality indicators
             Openness                             .134                 .307               .266               .392              .309              .201
                 Intellect                               .052                 .069               .465               .339              .133              .051
            Divergent thinking dimensions
             Fluency                               .150                 .168               .147               .171              .309              .254
              Elaboration                            .080                 .155               .029               .191              .220              .145

            Note.  TASA    Touchstone Applied Science Associates; LSA    Latent Semantic Analysis; EN    English; GloVe    Global Vectors for Word
            Representation; Word2Vec    word-to-vector; CFA   confirmatory factor analysis; MFRM   Many-Facet Rasch Model.
            p    .05.    p    .01.
<a id="page-15"></a>

### PDF 第 15 页

14                                DUMAS, ORGANISCIAK, AND DOHERTY


             activities of participants. Among creative personality indicators,      part, by the specific training and feedback that we provided our
         Openness significantly and positively predicted Arts, Cooking,      raters. In this case, raters were specifically trained to conceptualize
            Crafts, and Literary activities, whereas Intellect predicted Cooking     the Originality scale along which they rated AUT responses (that
         and Craft activities. Ideational Fluency significantly predicted both    ranged from 0 to 4) as a continuous dimension, on which most
           Literary and Musical activities, but Elaboration did not signifi-     responses would have a moderate amount of Originality (i.e., a 1,
           cantly predict any of the included creative activities. Interestingly,     2, or 3), and only a few responses would fall on the extremes of the
         none of these creativity indicators were capable of significantly     scale (i.e., 0 or 4). Using this particular prompting, the participant-
           predicting Performance activities, although the low-prevalence of     level latent Originality scores we calculated were able to achieve
          Performance within this general sample (as opposed to a profes-    a high level of reliability. In future work, it should not be assumed
            sionally creative sample) likely limited the variance of this activity    a priori that the highest score reliabilities are possible with human
           scale and therefore precluded a significant correlation.                 raters as opposed to text-mining systems. In cases where the
                                                             human raters are not effectively trained or are less motivated, the
                               Discussion                               scores from human raters could actually be less reliable than those
                                                                    from text-mining systems.
          As far as we are aware, this investigation has been the first          broadly.
           within the creativity research literature to compare the ability ofpublishers.         multiple text-mining models to produce reliable and valid Origi-   GloVe 840B Is the Recommended Text-Mining System
allied          nalitysuch, thisscoresstudyon hasthe aAUT,numberas ofcomparedprincipalwithfindingshuman-raters.and specificAs    for Originality               disseminated                                             A major stated goal of this investigation was to identify theits        recommendations to forward to the creativity research community.
   be                                                                                 publicly available text-mining system that  is most capable ofof   to     Four of these principal findings are described in detail below.
one                                                                        producing reliable and valid Originality scores on the AUT. Across    not                                                                                  the stages of the current investigation, the GloVe 840B system
or is
      Human Raters Can Produce the Most Reliable            emerged as the best system to choose for producing Originality
    and     Originality Scores                                                scores in creativity research. The GloVe system generated the most
     user        Within the creativity literature, the perennially low level of     reliable scores within a CFA framework as indicated by coefficientAssociation                                                                   H, and the second most reliable composite scores as indicated by           exact agreement (and therefore low level of interrater reliability)
                                                                    Alpha and Omega. The only system that produced more reliable          between human raters on the level of Originality of a given AUT
                                                                         composite scores  (i.e., TASA LSA) demonstrated substantially             individual      response has led many to bemoan the possibility of producing
                                                                      worse model-data-fit of the unidimensional CFA model as indi-    the      highly reliable creativity research using human raters (see Storme,
                                                                               cated by RMSEA. In addition, the TASA LSA scores correlatedPsychological of     Myszkowski, Çelik, & Lubart, 2014 for one approach to improving
                                                                                     substantially weaker with human-rated Originality than did the    use       this reliability). However, our results show that, if researchers are
                                                              GloVe scores, a strong indication that GloVe scores are more valid           willing to conceptualize human-rated Originality codes as ordinal
                                                                             than TASA scores. In addition, although both TASA and GloVeAmerican         indicators of an underlying Originality continuum and therefore
                                                                                scores did display the previously described potentially problematicthe  personal      average the Originality codes across raters, a very high level of
by the       reliability is possible with four trained raters. Of course, this high     relation to Elaboration (Forthmann et al., 2019), so too did the
    for       level of reliability refers to the consistency of the continuous     Originality scores coded by human-raters. The GloVe scores dis-
           Originality scores that are created by aggregating all 10 of the    played the correlation with Elaboration that was most in line with
                                                                                  the human-raters (although the difference between GloVe and        solely    AUT prompts included in this study (either by summing or throughcopyrighted        a CFA), and not to the individual AUT responses that were coded   TASA in this respect was not great). Further, among the text-
is         separately by each human rater. If an analysis at the fine-grained    mining systems, the GloVe scores had the correlations to other          intended       level of an individual response to a specific AUT prompt was     creative indicators (e.g., Openness and Intellect) that were most
   is      desired by a researcher, then interrater reliability may be a more     similar to that of human raters, although it should be noted thatdocument           informative index. Although such fine-grained analysis  at the    none of the Originality scoring systems used here (humans or         article           individual response level is somewhat common among creativity     text-mining) was significantly correlated with these indicators ofThis
     This      researchers with a basic psychology focus (Benedek, 2018), those     creative  activity, with the exception of a weak-moderate and
           researchers whose work is more situated within an applied psy-     negative correlation between the human-rated Originality scores
          chology area (e.g., educational psychology; Kerr & Stull, 2019)    and Cooking activities. Such a general lack of covariance in this
           are more commonly concerned with the capacity of creativity     regard may  be  caused by  the  psychological  differences  in
          measures to produce reliable and valid scores for participant-level     creativity-related self-report variables and more objectively quan-
            interpretation across multiple items, tasks, or prompts. Following      tified DT performance tasks such as the AUT. So, the near-zero
          with that applied-psychology focus, should a researcher or practi-     correlations from GloVe-based Originality scores to the Openness
            tioner desire to use the AUT to produce participant Originality    and Intellect measures are here interpreted as a positive finding
           scores for admission into a specialized educational program or     related to the validity of GloVe-based Originality scoring: GloVe
           personnel selection in the workplace,  it does appear from these    was capable of producing Originality scores that generally mim-
            results that four trained raters can produce highly reliable Origi-     icked the  criteria  correlations of the human-rated  Originality
            nality scores across 10 AUT prompts, at least with this general-     scores, but much more quickly and at a greatly reduced cost.
           population adult sample. Of course, the high level of reliability       In past investigations of text-mining models to produce Origi-
          achieved here by the human-raters was likely influenced, at least in     nality scores, by far the most commonly utilized text-mining
<a id="page-16"></a>

### PDF 第 16 页

MEASURING ORIGINALITY                                      15


          system has been the TASA system (e.g., Dumas, 2018), with a     the semantic richness of the ideas associated with that prompt. In
           small minority of other pieces utilizing the EN100k LSA system     essence, the findings of any psychometric research are tied to the
          (Forthmann et al., 2019). Given the results of this investigation,     actual item-stimuli that is administered to participants. Therefore, the
           researchers in this area should likely pivot their methodological     findings of the current study are most relevant for those researchers
           focus away from LSA based systems to GloVe, to create more    who administer the same or similar AUT prompts as we administered
            reliable and valid scores for research. GloVe’s improved reliability     here, and those researchers who plan to administer AUT prompts that
         and validity is likely attributable simultaneously to three factors:     are highly different than those administered here may need to interpret
           the size of its training corpus, the domain-generality of its training     our results with caution.
           corpus, and its probabilistic modeling approach. In general, such a      However, this lack of standardization of the AUT may also be an
          massive corpus (840 billion words, with 2.2 million unique words)     unexpected boon for the creativity literature, in that the AUT measure
        may be more capable of approximating the actual semantic struc-     or administration procedures can be easily updated as more psycho-
            ture of language-use than a smaller corpus (e.g., TASA’s unique     metric evidence becomes available. In this study, we found substantial
         words are only 4% of GloVe’s), leading to more psychologically     differences in the way individual AUT prompts (e.g., Book) contrib-
            reliable and valid Originality scores. Further, the inclusion of text     uted to the reliability of the latent Originality factor. For example, for
         from sources like Wikipedia, a general web crawl, and the British     both the human-rated and GloVe Originality scores, “Rope” was the          broadly.
           National Corpus make GloVe much more general in scope than the    weakest loading item, indicating that item detracts from the overallpublishers.      TASA corpus that is composed of only educational texts. Finally,      reliability of the Originality scores. However, this effect was more
allied        althoughdistance (orbothcosineGloVesimilarity)and LSAbetweenseek to wordrepresentvectors,the LSAeuclideanesti-    pronouncedraters were capablein GloVeof thanprovidingfor therelativelyhumans,stableimplyingestimatesthat theof humanpartic-               disseminated
its        mates these word vectors using a traditional parametric approach     ipant responses for “Rope,” whereas GloVe struggled more to quan-   be
of   to     and GloVe uses a more modern log-linear, or probabilistic method      tify the relevant semantic relations for that prompt. In addition, the
one          that may produce more psychologically relevant results (Penning-     loading for “Shovel” on the GloVe Originality factor was also rela-    not
or is      ton et al., 2014).                                                            tively weak, implying that text-mining system is worse at representing
    and       Of course, all of the text-mining models compared here (i.e.,     the semantic relations around “Shovel” than it is for the semantic
       TASA LSA, EN100k, word2vec, GloVe) are members of a larger     relations around “Brick,” for example. Overall, it is clear from these
     user      family of dimension-reduction–based techniques for the quantifi-     findings that the text-mining–based scoring methods had much moreAssociation
           cation of semantic relations among words and phrases through the     varying loadings across the AUT prompt than did the human-raters,
          examination of the angles among estimated word vectors. As    which illustrates the need, for those researchers who use text-mining             individual      previously reviewed, an alternative approach would be to quantify     systems to measure Originality, to choose their AUT prompts care-
    the      the semantic distance among words using a network science ap-      fully, and possibly pilot them with their chosen text-mining systemPsychological of      proach (De Deyne et al., 2016), in which the semantic distance is     before administering them to a large number of participants.
    use      not operationalized based on vector angles, but instead based on
           the path length between two or more words in the network. Some                                                                 Multiple Dimensions of Creativity AreAmerican        evidence in cognitive psychology suggests that the network sci-
                                                         Needed for Researchthe  personal      ence  approach may  have  advantages above  the  dimension-
by the      reduction approach when studying fine-grain cognitive processes       Although this current investigation was mainly focused on the
    for        (e.g., priming; Kumar et al., 2019). However, it remains a future     psychometric quality of Originality scores, the validity portion of the
        solely      directionbe helpfultoinascertainpsychometricwhetherworkthe networkthat, as withsciencetheapproachcurrent study,could     studyother alsodimensionsoffers someof DTvaluable(i.e., insightFluencyintoandtheElaboration),interrelationscreativeamongcopyrighted        aims to quantify Originality at the participant level with reliable     personality (i.e., Openness and Intellect), and particular real-world
is        and valid scores. Recent arguments in the creativity literature     creative activities. Although the AUT-based Originality scores did not          intended      (Kenett, 2019) suggest that the network science approach to quan-     well-predict self-reported creative activities, other dimensions of the
   is       tifying semantic distance may be fruitfully applied to computa-   AUT scoring (i.e., Fluency and Elaboration) did. For example, thedocument            tional psychometrics of creativity, and this approach may have     capability of Fluency scores to significantly and positively predict         articleThis        promise for creativity researchers.                                           literary and musical creative activities, and the prediction of arts
     This                                                                                           activities with AUT Elaboration, highlight the continued usefulness of
                                                                             both AUT Fluency and Elaboration scores in the creativity literature.        Not All AUT Prompts Contribute
                                                                      However, creative personality indicators were even better able to
         Equally to Reliability
                                                                                       predict self-reported creative activities, with Openness being the most
          One interesting, and perhaps problematic, aspect of creativity re-     predictive (i.e., significant positive correlations with Arts, Cooking,
           search is that the field’s most commonly used measure (i.e., the AUT)     Crafts, and Literary Activities) and Intellect also being reasonably
              is not necessarily fully standardized across studies in its administra-     predictive  (i.e., significant positive correlations with Cooking and
            tion procedures or even the particular prompts that are included. In     Crafts activities). Of course, given the self-report nature of the Cre-
             this study, the particular measure-administration choices, as well as     ative Activities Inventory as well as the personality questionnaires,
            the particular prompts included on the AUT, may mean that the      their relations may be inflated by participants’ creative self-concepts
             results could differ from other investigations where different measure-    (Karwowski, 2016). But, such an explanation can be ruled out for
         ment procedures were used. In addition, recent research (i.e., Beaty,    Fluency and Elaboration, both of which predicted creative activities as
           Kenett, Hass, & Schacter, 2019) has shown that certain prompts may     well as or stronger than human-rated Originality. Perhaps most im-
          be more facilitative of Ideational Fluency or Originality, depending on     portantly, it should be observed that, of the creative activities that
<a id="page-17"></a>

### PDF 第 17 页

16                                DUMAS, ORGANISCIAK, AND DOHERTY


          were significantly predicted in this study, none was predicted by all of      Memory & Cognition, 42, 1186–1197. http://dx.doi.org/10.3758/
            the creativity indicators. This finding strongly highlights the need for       s13421-014-0428-8
            researchers to measure a variety of different indicators of creative     Benedek, M. (2018). Internally directed attention in creative cognition. In
           potential—including both DT and personality assessments—to max-       R. E. Jung & O. Vartanian (Eds.), The Cambridge handbook of the
           imize the impact of our research to understand the creative process.       neuroscience of creativity (pp. 180–194). Cambridge, UK: Cambridge
        As a future direction in this line of investigation, it may be important        University Press. http://dx.doi.org/10.1017/9781316556238.011
            for researchers interested in the psychometrics of creativity to con-     Bianchi, F., & Palmonari, M. (2017). Joint learning of entity and type
                                                                               embeddings for analogical reasoning with entities. NL4AI@ AI  IA,            sider ways not only to automate Originality scoring using text-mining
                                                                                57–68.
           or other computational methods, but also to automate scoring systems
                                                                                          Bråten,  I., Ferguson, L. E., Strømsø, H.  I., & Anmarkrud, Ø. (2014).
            for other dimensions of divergent thinking and creative potential. For
                                                                                         Students working with multiple conflicting documents on a scientific
          example, it remains to be seen whether or how text-mining systems
                                                                                                     issue: Relations between epistemic cognition while reading and sourcing
          can be used for the quantification of Flexibility on the AUT or other                                                                                and argumentation in essays. British Journal of Educational Psychology,
       DT measures. In addition, creativity researchers have been perennially                                                                                            84, 58–85. http://dx.doi.org/10.1111/bjep.12005

           of their sexual or violent content (Dumas & Strickland, 2018; Hudson,      An investigation of corpus size and meaning in both latent semantic          broadly.       interested in participant responses to the AUT that stand out because     Crossley, S., Dascalu, M., & McNamara, D. (2017). How important is size?
           1968), and text-mining models could conceivably be applied to au-        analysis and latent Dirichlet allocation. Marco Island, Florida: Thepublishers.         tomatically identify such responses. In our view, these future direc-        Thirtieth International Florida Artificial Intelligence Research Society
allied          tionsin thehelpcreativityillustrateresearchthe potentialarea.  of computational psychometric work       Conference.FLAIRS/FLAIRS19/paper/viewFile/18299/17416Retrieved from https://www.aaai.org/ocs/index.php/               disseminated
its                                                              De Deyne, S., Verheyen, S., & Storms, G. (2016). Structure and organi-
   be
of   to                                                                                         zation of the mental lexicon: A network approach derived from syntactic
one      Coda                                                                dependency relations and word associations. In A. Mehler, A. Lücking,    not
or is         In our view, one main methodological bottleneck that limits the        S. Banisch, P. Blanchard, & B. Job (Eds.), Towards a theoretical
    and       productivity and impact of creativity research has historically been the       framework for analyzing complex linguistic networks (pp. 47–79). Ber-
                                                                                                                      lin, Germany: Springer. http://dx.doi.org/10.1007/978-3-662-47238-5_3
     user      time- and resource-intensiveness of human-rated DT tasks. We in the     Deerwester, S., Dumais, S. T., Furnas, G. W., Landauer, T. K., & Harsh-
             field have relied on hiring, training, and compensating human-ratersAssociation                                                                            man, R. (1990). Indexing by latent semantic analysis. Journal of the
            for decades, and graduate students situated within creativity research
                                                                               American Society for Information Science, 41, 391–407. http://dx.doi
            laboratories have also often shouldered the burden of rating hundreds                                                                                      .org/10.1002/(SICI)1097-4571(199009)41:6  391::AID-ASI1  3.0.CO;             individual      or thousands of DT responses for Originality. In some cases, very                                                                                      2-9
    the       large–scale studies of creativity may even have seemed infeasible    DeYoung, C. G., Quilty, L. C., & Peterson, J. B. (2007). Between facetsPsychological of      because of the burden of using human-raters. Here, we found the       and domains: 10 aspects of the Big Five. Journal of Personality and
    use     GloVe system is highly capable of approximating Originality scores        Social Psychology, 93, 880–896. http://dx.doi.org/10.1037/0022-3514
          produced via human-raters, but much more rapidly and potentially        .93.5.880American          free of cost. Based on the findings of this study, we may be nearing     Diedrich, J., Jauk, E., Silvia, P. J., Gredlein, J. M., Neubauer, A. C., &the  personal      a time in the field when the work of scoring DT tasks for Originality       Benedek, M. (2018). Assessment of real-life creativity: The Inventory of
by the      can be automated using a text-mining model, opening the door for        Creative Activities and Achievements (ICAA). Psychology of Aesthet-
    for     much larger-scale studies of DT and creativity, and hopefully leading          ics, Creativity, and the Arts, 12, 304–316. http://dx.doi.org/10.1037/
        solely       tosuchincreasedtext-miningreachsystemsand scopemayofberesearcheven easierin thetofield.run (e.g.,In thethroughfuture,    Dumas,aca0000137D. (2018). Relational reasoning and divergent thinking: An exam-copyrighted         more user-friendly software) and may contribute to a streamlined and        ination of the threshold hypothesis with quantile regression. Contempo-
is                                                                                     rary Educational Psychology, 53, 1–14. http://dx.doi.org/10.1016/j           automatic process of Originality measurement in creativity research.          intended                                                                                  .cedpsych.2018.01.003
   is                                                                    Dumas, D., & Dunbar, K. N. (2014). Understanding fluency and original-document                            References                                              ity: A latent variable perspective. Thinking Skills and Creativity, 14,         article                                                                         56–67. http://dx.doi.org/10.1016/j.tsc.2014.09.003This         Acar, S., & Runco, M. A. (2019). Divergent thinking: New methods, recent
     This          research, and extended theory. Psychology of Aesthetics, Creativity, and    Dumas, D., & Dunbar, K. N. (2016). The creative stereotype effect. PLoS                                                                    ONE, 11, e0142567. http://dx.doi.org/10.1371/journal.pone.0142567
               the Arts, 13, 153–158. http://dx.doi.org/10.1037/aca0000231
                                                                         Dumas, D., & Runco, M. (2018). Objectively scoring divergent thinking
            Barbot, B. (2018). The dynamics of creative ideation: Introducing a new
                                                                                                            tests for originality: A re-analysis and extension. Creativity Research
              assessment paradigm. Frontiers in Psychology, 9, 2529. http://dx.doi
                                                                                           Journal, 30, 466–468.
              .org/10.3389/fpsyg.2018.02529
                                                                         Dumas, D., & Strickland, A. L. (2018). From book to bludgeon: A closer
            Barbot, B., Tan, M., Randi, J., Santa-Donato, G., & Grigorenko, E. L.
                                                                                      look at unsolicited malevolent responses on the alternate uses task.               (2012). Essential skills for creative writing: Integrating multiple domain-
                                                                                                Creativity Research Journal, 30, 439–450. http://dx.doi.org/10.1080/                specific perspectives. Thinking Skills and Creativity, 7, 209–223. http://
               dx.doi.org/10.1016/j.tsc.2012.04.006                                     10400419.2018.1535790
            Beaty, R. E., Kenett, Y. N., Hass, R., & Schacter, D. L. (2019). A fan effect      Fleiss, J. L., & Cohen, J. (1973). The equivalence of weighted kappa and
                for creative thought: Semantic richness facilitates idea quantity but        the intraclass correlation coefficient as measures of reliability. Educa-
               constrains idea quality. PsyArXiv. Retrieved from https://psyarxiv.com/        tional and Psychological Measurement, 33(3), 613–619. http://dx.doi
              pfz2g/                                                                .org/10.1177/001316447303300309
            Beaty, R. E., Silvia, P. J., Nusbaum, E. C., Jauk, E., & Benedek, M. (2014).      Foltz, P. W., Streeter, L. A., Lochbaum, K. E., & Landauer, T. K. (2013).
            The roles of associative and executive processes in creative cognition.       Implementation and applications of the intelligent essay assessor. In
<a id="page-18"></a>

### PDF 第 18 页

MEASURING ORIGINALITY                                      17


           M. D. Shermis &  J. Burstein (Eds.), Handbook of automated essay        Aesthetics, Creativity, and the Arts, 12, 144–156. http://dx.doi.org/10
               evaluation: Current applications and new directions (pp. 68–88). New       .1037/aca0000125
             York, NY: Routledge/Taylor & Francis Group.                            Hirschberg, J., & Manning, C. D. (2015). Advances in natural language
             Forster, E. A., & Dunbar, K. N. (2009). Creativity evaluation through latent        processing. Science, 349, 261–266. http://dx.doi.org/10.1126/science
              semantic analysis. Proceedings of the Annual Conference of the Cogni-       .aaa8685
                 tive Science Society, 2009, 602–607.                                    Hocevar, D. (1980). Intelligence, divergent thinking, and creativity. Intel-
           Forthmann, B., Oyebade, O., Ojo, A., Günther, F., & Holling, H. (2019).        ligence, 4, 25–40. http://dx.doi.org/10.1016/0160-2896(80)90004-5
              Application of latent semantic analysis to divergent thinking is biased by     Hornberg,  J., & Reiter-Palmon, R. (2017). Creativity and the big five
                elaboration. The Journal of Creative Behavior. Advance online publi-        personality traits: Is the relationship dependent on the creativity mea-
                cation. http://dx.doi.org/10.1002/jocb.240                                     sure? In G. J. Feist, R. Reiter-Palmon, & J. C. Kaufman (Eds.), The
           Forthmann, B., Szardenings, C., & Holling, H. (2020). Understanding the      Cambridge handbook of creativity and personality research (pp. 275–
             confounding effect of fluency in divergent thinking scores: Revisiting        293). New York, NY: Cambridge University Press. http://dx.doi.org/10
              average scores to quantify artifactual correlation. Psychology of Aesthet-       .1017/9781316228036.015
                  ics, Creativity, and the Arts. Advance online publication. http://dx.doi     Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance
              .org/10.1037/aca0000196                                                         structure analysis: Conventional criteria versus new alternatives. Struc-
           Forthmann, B., Wilken, A., Doebler, P., & Holling, H. (2019). Strategy        tural Equation Modeling,  6, 1–55.  http://dx.doi.org/10.1080/          broadly.
               induction enhances creativity in figural divergent thinking. The Journal       10705519909540118
                of Creative Behavior, 53, 18–29. http://dx.doi.org/10.1002/jocb.159       Hudson, L. (1968). Frames of mind: Ability, perception and self-perceptionpublishers.
           Furnham, A., Crump, J., & Swami, V. (2009). Abstract reasoning and big        in the arts and sciences. Oxford, England: Norton.
                five personality correlates of creativity in a British occupational sample.allied                                                                           Karwowski, M. (2016). The dynamics of creative self-concept: Changes               disseminated         Imagination, Cognition and Personality, 28, 361–370. http://dx.doi.org/
its                                                                              and reciprocal relations between creative self-efficacy and creative per-
   be         10.2190/IC.28.4.fof                                                                                         sonal identity. Creativity Research Journal, 28, 99–104. http://dx.doi
   to      Gray, K., Anderson, S., Chen, E. E., Kelly, J. M., Christian, M. S., Patrick,                                                                                     .org/10.1080/10400419.2016.1125254one    not             J., . . . Lewis, K. (2019). “Forward flow”: A new measure to quantify     Karwowski,                                                                                            M., & Gralewski, J. (2013). Threshold hypothesis: Fact or
or is          free thought and predict creativity. The American Psychologist, 74,                                                                                                         artifact? Thinking Skills and Creativity, 8, 25–33. http://dx.doi.org/10
    and        539–554. http://dx.doi.org/10.1037/amp0000391                               .1016/j.tsc.2012.05.003
           Green, A. E., Kraemer, D. J. M., Fugelsang, J. A., Gray, J. R., & Dunbar,
     user        K. N. (2010). Connecting long distance: Semantic distance in analogical     Kenett, Y. N. (2019). What can quantitative measures of semantic distance
                                                                                                                        tell us about creativity? Current Opinion in Behavioral Sciences, 27,Association            reasoning modulates frontopolar cortex activity. Cerebral Cortex, 20,
                                                                                    11–16. http://dx.doi.org/10.1016/j.cobeha.2018.08.010
             70–76. http://dx.doi.org/10.1093/cercor/bhp081
                                                                                          Kenett, Y. N., Levi, E., Anaki, D., & Faust, M. (2017). The semantic
           Green, A. E., Kraemer, D. J. M., Fugelsang, J. A., Gray, J. R., & Dunbar,             individual                                                                                        distance task: Quantifying semantic distance with semantic network path
             K. N. (2012). Neural correlates of creativity in analogical reasoning.
    the                                                                                            length. Journal of Experimental Psychology: Learning, Memory, and              Journal of Experimental Psychology: Learning, Memory, and Cogni-Psychological of                                                                                     Cognition, 43, 1470–1489. http://dx.doi.org/10.1037/xlm0000391
                 tion, 38, 264–272. http://dx.doi.org/10.1037/a0025764
    use       Guilford, J. P. (1967). The nature of human intelligence. New York, NY:     Kerr, B. A., & Stull, O. A. (2019). Measuring creativity in research and
                                                                                                     practice. In M. W. Gallagher & S. J. Lopez (Eds.), Positive psycholog-
             McGraw-Hill.
                                                                                                      ical assessment: A handbook of models and measures (2nd ed., pp.American         Günther, F., Dudschig, C., & Kaup, B. (2015). LSAfun-An R package for
                                                                                  125–138). Washington, DC: American Psychological Association.the  personal         computations based on Latent Semantic Analysis. Behavior Research
                                                                                           http://dx.doi.org/10.1037/0000138-009
by the        Methods, 47, 930–944. http://dx.doi.org/10.3758/s13428-014-0529-0
                                                                                         Kintsch, W., & Bowles, A. R. (2002). Metaphor comprehension: What
    for      Gwet, K. L. (2014). Handbook of inter-rater reliability, 4th ed.: The
                                                                           makes a metaphor difficult to understand? Metaphor and Symbol, 17,                definitive guide to measuring the extent of agreement among raters.
                                                                               249–262. http://dx.doi.org/10.1207/S15327868MS1704_1               Gaithersburg, MD: Advanced Analytics, LLC.        solely                                                                                              Kjell, O. N. E., Kjell, K., Garcia, D., & Sikström, S. (2019). Semanticcopyrighted         Hancock, G. R. (2001). Effect size, power, and sample size determination
                                                                                      measures: Using natural language processing to measure, differentiate,is             for structured means modeling and mimic approaches to between-groups
                                                                                and describe psychological constructs. Psychological Methods, 24, 92–              hypothesis testing of means on a single latent construct. Psychometrika,          intended
   is         66, 373–388. http://dx.doi.org/10.1007/BF02294440                         115. http://dx.doi.org/10.1037/met0000191
            Hass, R. W. (2017a). Semantic search during divergent thinking. Cogni-     Kuhn, J. T., & Holling, H. (2009). Measurement invariance of divergentdocument
                 tion, 166, 344–357. http://dx.doi.org/10.1016/j.cognition.2017.05.039          thinking across gender, age, and school forms. European Journal of         articleThis         Hass, R. W. (2017b). Tracking the dynamics of divergent thinking via       Psychological Assessment, 25, 1–7. http://dx.doi.org/10.1027/1015-5759
     This         semantic distance: Analytic methods and theoretical implications. Mem-        .25.1.1
              ory & Cognition, 45, 233–244. http://dx.doi.org/10.3758/s13421-016-    Kumar, A. A., Balota, D. A., & Steyvers, M. (2019). Distant connectivity
             0659-y                                                             and multiple-step priming in large-scale semantic networks. Journal of
           He, Q., von Davier, M., Greiff, S., Steinhauer, E. W., & Borysewicz, P. B.       Experimental Psychology: Learning, Memory, and Cognition. Advance
               (2017). Collaborative problem solving measures in the Programme for        online publication. http://dx.doi.org/10.1037/xlm0000793
                International Student Assessment (PISA). In A. A. von Davier, M. Zhu,     Landauer, T. K., & Dumais, S. T. (1997). A solution to Plato’s problem:
        & P. C. Kyllonen (Eds.), Innovative assessment of collaboration (pp.      The latent semantic analysis theory of acquisition, induction, and rep-
              95–111). Cham, Switzerland: Springer. http://dx.doi.org/10.1007/978-3-        resentation of knowledge. Psychological Review, 104, 211–240. http://
             319-33261-1_7                                                           dx.doi.org/10.1037/0033-295X.104.2.211
           Hedge, C., Powell, G., & Sumner, P. (2018). The reliability paradox: Why     Landauer, T. K., Foltz, P. W., & Laham, D. (1998). An introduction to
               robust cognitive tasks do not produce reliable individual differences.         latent semantic analysis. Discourse Processes, 25, 259–284. http://dx
             Behavior Research Methods, 50, 1166–1186. http://dx.doi.org/10.3758/       .doi.org/10.1080/01638539809545028
             s13428-017-0935-1                                                    Landauer, T. K., Laham, D., Rehder, B., & Schreiner, M. E. (1997). How
           Heinen, D. J. P., & Johnson, D. R. (2018). Semantic distance: An auto-        well can passage meaning be derived without using word order? A
             mated measure of creativity that is novel and appropriate. Psychology of       comparison of Latent Semantic Analysis and humans. Proceedings of
<a id="page-19"></a>

### PDF 第 19 页

18                                DUMAS, ORGANISCIAK, AND DOHERTY


               the 19th Annual Meeting of the Cognitive Science Society, 412–417.         alization. Proceedings of the Second AAAI Conference on Human Compu-
           Mahwah, NJ: Erlbaum.                                                             tation and Crowdsourcing (HCOMP 2014). Retrieved from https://www
           Landauer, T. K., McNamara, D. S., Dennis, S., & Kintsch, W. (2013).       .aaai.org/ocs/index.php/HCOMP/HCOM-P14/paper/viewFile/8972/
           Handbook of latent semantic analysis. London, UK: Psychology      8969
               Press.                                                                  Pennington, J., Socher, R., & Manning, C. (2014, October). Glove: Global
            Linacre, J. M., & Wright, B. D. (1988). Facets. Chicago, IL: MESA.            vectors for word representation. In A. Moscitti, A. Pang, & B. Dael-
            Lord, F. M. (2012). Applications of item response theory to practical      emans (Eds.), Proceedings of the 2014 conference on empirical methods
                testing problems. London, UK: Routledge. http://dx.doi.org/10.4324/        in natural language processing (pp. 1532–1543). Doha, Qatar: Associ-
            9780203056615                                                                ation for Computational Linguistics. http://dx.doi.org/10.3115/v1/D14-
           Marek, R. J., & Ben-Porath, Y. S. (2017). Using the Minnesota Multiphasic      1162
               Personality Inventory-2-Restructured Form (MMPI-2-RF) in behavioral      Piatelli-Palmarini, M. (1980). Language and learning: The debate between
             medicine settings. In M. E. Maruish (Ed.), Handbook of psychological       Jean Piaget and Noam Chomsky. Cambridge, MA: Harvard University
              assessment in primary care settings (pp. 631–662, 2nd ed.). New York,        Press.
           NY: Routledge/Taylor & Francis Group.                                   Plucker, J. A., & Makel, M. C. (2010). Assessment of creativity. In J. C.
          McDonald, R. P. (1999). Test theory: A unified approach. Mahwah, NJ:      Kaufman & R. J. Sternberg (Eds.), The Cambridge handbook of cre-          broadly.        Erlbaum.                                                                                ativity (pp. 48–73). New York, NY: Cambridge University Press. http://
         McKay, A. S., Karwowski, M., & Kaufman, J. C. (2017). Measuring the       dx.doi.org/10.1017/CBO9780511763205.005publishers.           muses: Validating the Kaufman Domains of Creativity Scale (K-DOCS).     Prabhakaran, R., Green, A. E., & Gray,  J. R. (2014). Thin slices of
allied           Psychologydx.doi.org/10.1037/aca0000074of Aesthetics, Creativity, and the Arts, 11, 216–230. http://         creativity:Behavior ResearchUsing single-wordMethods, 46,utterances641–659.to assesshttp://dx.doi.org/10.3758/creative cognition.               disseminated
its        McNeish, D. (2018). Thanks coefficient alpha, we’ll take  it from here.       s13428-013-0401-7
   be
of   to        Psychological Methods, 23, 412– 433. http://dx.doi.org/10.1037/     Primi, R., Silvia, P. J., Jauk, E., & Benedek, M. (2019). Applying many-
one          met0000144                                                                      facet Rasch modeling in the assessment of creativity. Psychology of    not
or is      McNeish, D., An,  J., & Hancock, G. R. (2018). The thorny relation        Aesthetics, Creativity, and the Arts, 13, 176–186. http://dx.doi.org/10
    and        between measurement quality and  fit index cutoffs in latent variable       .1037/aca0000230
              models. Journal of Personality Assessment, 100, 43–52. http://dx.doi     Puryear, J. S., Kettler, T., & Rinn, A. N. (2017). Relationships of person-
     user         .org/10.1080/00223891.2017.1281286                                               ality to differential conceptions of creativity: A systematic review.Association        McNeish, D., & Wolf, M. G. (2020). Thinking twice about sum scores.       Psychology of Aesthetics, Creativity, and the Arts, 11, 59–68. http://dx
             Behavior Research Methods. Advanced online publication. http://dx.doi       .doi.org/10.1037/aca0000079
              .org/10.3758/s13428-020-01398-0                                        Reiter-Palmon, R., Forthmann, B., & Barbot, B. (2019). Scoring divergent             individual
           Mednick, S. A. (1962). The associative basis of the creative process.        thinking  tests: A review and systematic framework. Psychology of
    the             Psychological Review, 69, 220 –232. http://dx.doi.org/10.1037/        Aesthetics, Creativity, and the Arts, 13, 144–152. http://dx.doi.org/10Psychological of
            h0048850                                                             .1037/aca0000227
    use      Messick, S. (1995). Validity of psychological assessment: Validation of     Robertson, S. E., & Jones, K. S. (1976). Relevance weighting of search
               inferences from persons’ responses and performances as scientific in-        terms. Journal of the American Society for Information Science, 27,American               quiry into score meaning. American Psychologist, 50, 741–749. http://       129–146. http://dx.doi.org/10.1002/asi.4630270302the  personal         dx.doi.org/10.1037/0003-066X.50.9.741                                         Silvia, P. J., Winterstein, B. P., Willse, J. T., Barona, C. M., Cram, J. T.,
by the      Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation       Hess, K.  I.,  .  .  . Richard, C. A. (2008). Assessing creativity with
    for          of word representations in vector space. arXiv preprint arXiv,1301.        divergent thinking tasks: Exploring the reliability and validity of new
        solely      Mikolov,3781. RetrievedT., Sutskever,from https://arxiv.org/abs/1301.3781I., Chen, K., Corrado, G. S., & Dean, J. (2013).        subjectivethe Arts, 2,scoring68–85.methods.http://dx.doi.org/10.1037/1931-3896.2.2.68Psychology of Aesthetics, Creativity, andcopyrighted             Distributed representations of words and phrases and their composition-    Snow, R., O’Connor, B., Jurafsky, D., & Ng, A. Y. (2008, October). Cheap
is               ality. In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, & K. Q.       and fast—But is it good?: evaluating non-expert annotations for natural          intended        Weinberger (Eds.), Advances in neural information processing systems       language tasks. Proceedings of the Conference on Empirical Methods in
   is          (pp. 3111–3119). Red Hook, NY: Curran Associates, Inc.                   Natural Language Processing (pp. 254–263). Stroudsburg, PA: Asso-document         Muthén, L. K., & Muthén, B. (2019). Mplus user’s guide (8th ed.). Los        ciation for Computational Linguistics.         article         Angeles, CA: Author.                                                               S¸tefa˘nescu, D., Banjade, R., & Rus, V. (2014, May). Latent semanticThis
     This      Myszkowski, N., & Storme, M. (2019). Judge response theory? A call to        analysis models on Wikipedia and TASA. The 9th Language Resources
             upgrade our psychometrical account of creativity judgments. Psychology       Evaluation Conference (LREC), Reykjavik, Iceland. Retrieved from
                of Aesthetics, Creativity, and the Arts, 13, 167–175. http://dx.doi.org/10        http://www.lrec-conf.org/proceedings/lrec2014/pdf/403_Paper.pdf
             .1037/aca0000225                                                     Storme, M., Çelik, P., Camargo, A., Forthmann, B., Holling, H., & Lubart,
            Oleynick, V. C., DeYoung, C. G., Hyde, E., Kaufman, S. B., Beaty, R. E.,        T. (2017). The effect of forced language switching during divergent
        & Silvia, P.  J. (2017). Openness/intellect: The core of the creative        thinking: A study on bilinguals’ originality of ideas. Frontiers in Psy-
                personality. In G. J. Feist, R. Reiter-Palmon, & J. C. Kaufman (Eds.),        chology, 8, 2086. http://dx.doi.org/10.3389/fpsyg.2017.02086
            The Cambridge handbook of creativity and personality research (pp.     Storme, M., Myszkowski, N., Çelik, P., & Lubart, T. (2014). Learning to
              9–27). New York, NY: Cambridge University Press. http://dx.doi.org/       judge creativity: The underlying mechanisms in creativity training for
             10.1017/9781316228036.002                                               non-expert judges. Learning and Individual Differences, 32, 19–25.
            Organisciak, P. (2016). Term weights for 235k language and literature        http://dx.doi.org/10.1016/j.lindif.2014.03.002
                 texts [Data set]. Retrieved from https://www.ideals.illinois.edu/handle/     Torrance, E. P. (1972). Predictive validity of the Torrance Tests of Creative
             2142/89691                                                               Thinking. The Journal of Creative Behavior, 6, 236–262. http://dx.doi
             Organisciak, P., Teevan, J., Dumais, S., Miller, R. C., & Kalai, A. T. (2014,        .org/10.1002/j.2162-6057.1972.tb00936.x
              September). A crowd of your own: Crowdsourcing for on-demand person-     Torrance, E. P. (1988). The nature of creativity as manifest in its testing.
<a id="page-20"></a>

### PDF 第 20 页

MEASURING ORIGINALITY                                      19


               In R. J. Sternberg (Ed.), The nature of creativity (pp. 43–75). New York,     Yoon, C. H. (2017). A validation study of the Torrance Tests of Creative
           NY: Cambridge University Press.                                        Thinking with a sample of Korean elementary school students. Thinking
          von Davier, A. A. (2017). Computational psychometrics in support of         Skills and Creativity, 26, 38–50. http://dx.doi.org/10.1016/j.tsc.2017.05
               collaborative educational assessments. Journal of Educational Measure-       .004
              ment, 54, 3–11. http://dx.doi.org/10.1111/jedm.12129
           White, H. A., & Shah, P. (2016). Scope of semantic activation and
               innovative thinking in college students with ADHD. Creativity Research                                       Received June 3, 2019
              Journal, 28, 275–282. http://dx.doi.org/10.1080/10400419.2016                             Revision received January 29, 2020
             .1195655                                                                                 Accepted February 19, 2020


          broadly.
publishers.
allied               disseminated
its
   be
of
   to
one    not
or is
    and
     userAssociation
             individual
    thePsychological of
American usethe  personal
by the
    for
        solelycopyrighted
is
          intended
   isdocument
         articleThis
     This
