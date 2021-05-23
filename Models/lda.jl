using Dates
using Flatten
using TopicModelsVB

pwd()

# read in LDA-c corpus
f = open("NLP_project/Models/lda_corpus.lda-c")
lda_corpus = readlines(f, keep=true);

# read time slices
f = open("NLP_project/Models/time_slices.txt")
time_slices = readlines(f, keep=false);
time_slices = parse.(Int, time_slices);

# get years
years = collect(2000:2021)

# get for each year the articles
year_articles = []
begin_n = 0
end_n = 0
for (index, value) in enumerate(time_slices)
    if index == 1
        begin_n = 1
        end_n = value
    else
        begin_n = end_n+1
        end_n = begin_n + value - 1
    end
    push!(year_articles, lda_corpus[begin_n:end_n])
end

# init final corpus
final_corpus = [string(length(time_slices))*'\n']

open("NLP_project/Models/test.txt", "w") do io
    write(io, string(length(time_slices))*'\n')

    for i in eachindex(year_articles)

        # append time stamp in unix time
        write(io, string(datetime2unix(DateTime(years[i])))*'\n')
    
        # append number of articles in time slice
        write(io, string(time_slices[i])*'\n')
    
        # append articles in usual lda-c format
        for article in year_articles[i]
            write(io, article)
        end
    end
end;

length(lda_corpus)


lda_corpus[1]


f = open("NLP_project/data/clean/lda_corpus.lda-c.vocab")
vocab = readlines(f, keep=true);

readcorp(docfile = "NLP_project/data/clean/lda_corpus.lda-c", vocabfile = "NLP_project/data/clean/lda_corpus.lda-c.vocab")

