import sys
import xml.etree.ElementTree as etree
import mwparserfromhell as parser
import time
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from py_stringmatching.similarity_measure.soft_tfidf import SoftTfIdf
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.jaro_winkler import JaroWinkler
from py_stringmatching.similarity_measure.monge_elkan import MongeElkan

class WikipediaNavigator(object):

    def __init__(self, path_to_file):
        self._encoding = "utf-8"
        self._pathToFile = path_to_file
        self._tags = {"Media:", "Special:", "Talk:", "User:", "User talk:", "Wikipedia:", "Wikipedia talk:", "File:",
                      "File talk:", "MediaWiki:", "MediaWiki talk:", "Template:", "Template talk:", "Help:", "Help talk:",
                      "Category:", "Category talk:", "Portal:", "Portal talk:", "Book:", "Book talk:", "Draft:", "Draft talk:",
                      "Education Program:", "Education Program talk:", "TimedText:", "TimedText talk:", "Module:", "Module talk:",
                      "Gadget:", "Gadget talk:", "Gadget definition:", "Gadget definition talk:", "Topic:"}


    def parallelizedTrainingSetCreation(self, infoboxType):
        totalCount = 0
        withInfobox = 0
        start_time = time.time()

        context = etree.iterparse(self._pathToFile, events=('start', 'end'))
        context = iter(context)
        event, root = context.next()

        for event, elem in context:
            tname = self.strip_tag_name(elem.tag)

            if event == 'start':

                if tname == 'page':
                    articleTitle = ''
                    text = ''

            elif event == 'end':
                if tname == 'title' and all(s not in tname for s in self._tags):
                    title = str(elem.text)
                    articleTitle = title.strip().rstrip().replace(" ","_")
                    elem.clear()

                elif tname == 'text':
                    text = elem.text

                    if '#REDIRECT' not in text:
                        try:
                            '''
                            USE MWPARSERFROMHELL TO CLEAN TEXT AND GET INFOBOX
                            '''
                            infoboxTemplateName, infoboxTuples = self.getInfoboxInfo(text)
                            cleanedText = self.getCleanedText(text)

                        except parser.parser.ParserError:
                            print("%s - ParseError mwparserfromhell could not parse wikicode - ignoring article" % articleTitle)
                            continue
                        elem.clear()

                elif tname == 'page':  # ends page
                    if cleanedText is not '' and "#REDIRECT" not in text \
                            and infoboxTemplateName is not '' and len(infoboxTuples) != 0:
                        withInfobox += 1

                        self.doSentenceMatching(cleanedText, infoboxTuples)

                    totalCount += 1

                    elem.clear()

                root.clear()

        elapsed_time = time.time() - start_time
        print("Total pages: {:,}".format(totalCount))
        print("Infoboxes: {:,}".format(withInfobox))
        print("Elapsed time: {}".format(self.elapsedTime(elapsed_time)))

    def doSentenceMatching(self, cleanedText, infoboxTuples):
        '''
                                DO SENTENCE MATCHING WITH INFOBOX TUPLES
                                SAVE DATASET TO FILE
                            '''
        tokenizedTuples = [word_tokenize(prop.replace("_", " ") + " " + value) for prop, value in infoboxTuples]
        corpus = copy.deepcopy(tokenizedTuples)
        # tokenize text into sentences
        sentences = sent_tokenize(cleanedText.replace("\n", ". ").replace("\t", ". ").strip().rstrip())
        for sentence in sentences:
            tokenizedSent = word_tokenize(sentence)
            corpus.append(tokenizedSent)
        soft_tfidf = SoftTfIdf(corpus, sim_func=JaroWinkler().get_raw_score, threshold=0.8)
        oc = OverlapCoefficient()
        # try to match each infobox tuple to sentences in text
        for tokenizedTuple in tokenizedTuples:
            print(tokenizedTuple)
            selectedSentences = []

            for sentence in sentences:
                tokens = word_tokenize(sentence)
                soft_raw_score = soft_tfidf.get_raw_score(tokens, tokenizedTuple)
                oc_raw_score = oc.get_raw_score(tokens, tokenizedTuple)

                if soft_raw_score > 0.1 or oc_raw_score >= 0.6:
                    selectedSentences.append((sentence, soft_raw_score, oc_raw_score))

            ordered = sorted(selectedSentences, key=lambda item: (item[2], item[1]), reverse=True)

            if len(ordered) > 0:

                selected = None
                for ranked in ordered:
                    if ranked[1] >= 0.05 and ranked[2] >= 0.5:
                        selected = ranked

                if selected != None:
                    print(">>>>> (%s, %s, %s)" % selected)

    def getCleanedText(self, text):
        keep = ['convert', 'airport codes']

        wikicode = parser.parse(text)
        templates = wikicode.filter_templates()

        # remove irrelevant templates
        for template in templates:
            try:
                if template.name.lower() not in keep or len(template.name.lower().rstrip().strip()) == 0:
                    if wikicode.contains(template):
                        wikicode.remove(template, recursive=True)
            except ValueError:
                continue

        # remove links for images and files
        wikilinks = wikicode.filter_wikilinks()
        for link in wikilinks:
            for tag in ['[file:', '[image:']:
                if tag in link.lower() and wikicode.contains(link):
                    wikicode.remove(link)

        # remove tables
        tables = wikicode.filter_tags(matches=lambda node: node.tag == 'table')
        for table in tables:
            if wikicode.contains(table):
                wikicode.remove(table)

        return wikicode.strip_code(collapse=True, keep_template_params=True)

    def getInfoboxInfo(self, text):
        infoboxTemplateName = ''
        infoboxTuples = []

        if len(text) > 3000:
            sampleText = text[0:2999]
            infoboxWikicode = parser.parse(sampleText)
        else:
            infoboxWikicode = parser.parse(text)

        templates = infoboxWikicode.filter_templates()
        # detects infobox template

        infoboxTemplate = None
        for template in templates:
            if 'infobox' in template.name.lower():
                infoboxTemplate = template

        # get infobox tuples in case infobox exists
        if infoboxTemplate is not None:
            infoboxTemplateName = infoboxTemplate.name

            if infoboxTemplate.params is not None:
                for p in infoboxTemplate.params:
                    try:
                        prop = p.name.rstrip().strip()
                        value = parser.parse(p.value)

                        # cleans property value in case of
                        # existing html tags or links
                        htmltags = value.filter_tags()
                        for tag in htmltags:
                            value.remove(tag, recursive=True)

                        templates = value.filter_templates()
                        for temp in templates:
                            if 'cite web' in temp.name.lower():
                                value.remove(temp, recursive=True)

                        value = value.strip_code(keep_template_params=True).strip().rstrip().replace("\n", ", ")
                        if len(value) == 0:
                            continue
                        else:
                            infoboxTuples.append([prop, value])
                    except ValueError:
                        continue

        return infoboxTemplateName, infoboxTuples

    def strip_tag_name(self, t):
        idx = k = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        return t

    def elapsedTime(self, sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)