# -*- coding: utf-8 -*-

from Bio import Entrez
from Bio import Medline


def search_and_save(email, query, filename, mindate, nrqueries):
    """
    Function for PubMed article search and save.

    :param email: the user email.
    :param query: the desired query (same as a normal PubMed search).
    :param filename: label for the identification of the articles.
    :param mindate: minimum search date.
    :param nrqueries: number of articles to retrieve.
    :return: saves a given number of articles to the root directory.
    """
    # Retrieve the query
    Entrez.email = email

    handle = Entrez.esearch(db="pubmed",
                            term=query,
                            datetype="pdat",
                            usehistory="y",
                            retmax=nrqueries,
                            MIN_DATE=mindate)

    record = Entrez.read(handle)

    idlist = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text", retmax=nrqueries, MIN_DATE=mindate)
    records = Medline.parse(handle)

    # Save the list with the documents and their information
    papers = []
    for record in records:
        papers.append(record)

    # Get the desired information from the query
    out = []
    paper_string = ""
    for paper in papers:
        try:
            paper_string = "{TI} \n {AB}".format(
                TI=paper['TI'],
                AB=paper['AB'])
        except KeyError:
            papers.pop(papers.index(paper))
        finally:
            out.append(paper_string)

    # Save a file in the same directory of this script with
    # the values specified (Authors, Title, Abstract)
    for i, name in enumerate(out):
        f = open(filename + "_" + str(i + 1) + ".txt", "w")
        f.write(name + "\n")
        f.close()


    print("Saving information for " + str(len(out)) + " out of " + str(len(papers)) + " articles.")
    print("Done!")



if __name__ == '__main__':

    print("\nStarting PubMed Search & Save. Please answer a few questions:")
    email = raw_input("What's your email? (NCBI requires you to specify your email address with each request) ")
    query = raw_input("Please enter the desired query: ")
    filename = raw_input("Please enter the desired label for the retrieved files: ")
    mindate = raw_input("What's the mininum date for search? (YYYY/MM/DD) ")
    nrqueries = raw_input("Finally, how many files do you wish to retrieve? ")
    print("\n")

    search_and_save(email, query, filename, mindate, nrqueries)
