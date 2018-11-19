package edu.vt.cs.ir.prj;

import edu.vt.cs.ir.search.AbstractQLSearcher;
import edu.vt.cs.ir.search.LuceneTermPostingList;
import edu.vt.cs.ir.search.PostingList;
import edu.vt.cs.ir.utils.*;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Created by liyanli on 11/14/18.
 */
public class RM3 extends AbstractQLSearcher {

    public static void main( String[] args ) {
        try {

            String pathIndex = "C:/Users/Liyan/Desktop/index_lucene_robust04_krovetz"; // change it to your own index path
            Analyzer analyzer = LuceneUtils.getAnalyzer( LuceneUtils.Stemming.Krovetz ); // change the stemming setting accordingly

            String pathQueries = "C:/Users/Liyan/Desktop/CS5604_FinalProject/queries_robust04"; // change it to your query file path
            String pathQrels = "C:/Users/Liyan/Desktop/CS5604_FinalProject/qrels_robust04"; // change it to your qrels file path
            String pathStopwords = "C:/Users/Liyan/Desktop/CS5604_FinalProject/stopwords_inquery"; // change to your stop words path

            String field_docno = "docno";
            String field_search = "content";

            RM3 searcher = new RM3( pathIndex );
            searcher.setStopwords( pathStopwords );

            Map<String, String> queries = EvalUtils.loadQueries( pathQueries );
            Map<String, Set<String>> qrels = EvalUtils.loadQrels( pathQrels );

            int top = 1000;
            double mu = 1000;
            int numfbterms = 100;
            double alpha = 0.2;
            
            double[] p10 = new double[queries.size()];
            double[] ap = new double[queries.size()];
            int ix = 0;
            for (String qid : queries.keySet()) {

                String query = queries.get(qid);
                List<String> terms = LuceneUtils.tokenize(query, analyzer);
                String[] termsarray = new String[terms.size()];
                for (int k = 0; k < termsarray.length; k++) {
                    termsarray[k] = terms.get(k);
                }

                //List<SearchResult> results = searcher.search( field_search, terms, mu, top );
                Map<String, Double> rm1 = searcher.estimateQueryModelRM1(field_search, terms, mu, mu, top, numfbterms);
                Map<String, Double> termWeights = searcher.estimateQueryModelRM3( terms, rm1, alpha );
                List<SearchResult> results = searcher.search(field_search, termWeights, mu, top);
                SearchResult.dumpDocno(searcher.index, field_docno, results);

                p10[ix] = EvalUtils.precision(results, qrels.get(qid), 10);
                ap[ix] = EvalUtils.avgPrec(results, qrels.get(qid), top);

                /*System.out.printf(
                        "%-10s%8.3f%8.3f\n",
                        qid,
                        p10[ix],
                        ap[ix]
                );*/
                ix++;
            }
            System.out.printf(
                    "%-10s%-25s%10.3f%10.3f%10.3f\n",
                    "QL",
                    "QL",
                    StatUtils.mean( p10 ),
                    StatUtils.mean( ap ),
                    alpha
            );



            searcher.close();

        } catch ( Exception e ) {
            e.printStackTrace();
        }
    }

    protected File dirBase;
    protected Directory dirLucene;
    protected IndexReader index;
    protected Map<String, DocLengthReader> doclens;

    public RM3( String dirPath ) throws IOException {
        this( new File( dirPath ) );
    }

    public RM3( File dirBase ) throws IOException {
        this.dirBase = dirBase;
        this.dirLucene = FSDirectory.open( this.dirBase.toPath() );
        this.index = DirectoryReader.open( dirLucene );
        this.doclens = new HashMap<>();
    }

    public IndexReader getIndex() {
        return this.index;
    }

    public PostingList getPosting(String field, String term ) throws IOException {
        return new LuceneTermPostingList( index, field, term );
    }

    public DocLengthReader getDocLengthReader( String field ) throws IOException {
        DocLengthReader doclen = doclens.get( field );
        if ( doclen == null ) {
            doclen = new DocLengthReader( this.dirBase, field );
            doclens.put( field, doclen );
        }
        return doclen;
    }

    public void close() throws IOException {
        index.close();
        dirLucene.close();
        for ( DocLengthReader doclen : doclens.values() ) {
            doclen.close();
        }
    }

    public Map<String, Double> estimateQueryModelRM1( String field, List<String> terms, double mu1, double mu2, int numfbdocs, int numfbterms ) throws IOException {

        List<SearchResult> results = search( field, terms, mu1, numfbdocs );
        Set<String> voc = new TreeSet<>();
        for ( SearchResult result : results ) {
            TermsEnum iterator = index.getTermVector( result.getDocid(), field ).iterator();
            BytesRef br;
            while ( ( br = iterator.next() ) != null ) {
                if ( !isStopwords( br.utf8ToString() ) ) {
                    voc.add( br.utf8ToString() );
                }
            }
        }

        Map<String, Double> collector = new HashMap<>();
        for ( SearchResult result : results ) {
            double ql = result.getScore();
            double dw = Math.exp( ql );
            TermsEnum iterator = index.getTermVector( result.getDocid(), field ).iterator();
            Map<String, Integer> tfs = new HashMap<>();
            int len = 0;
            BytesRef br;
            while ( ( br = iterator.next() ) != null ) {
                tfs.put( br.utf8ToString(), (int) iterator.totalTermFreq() );
                len += iterator.totalTermFreq();
            }
            for ( String w : voc ) {
                int tf = tfs.getOrDefault( w, 0 );
                double pw = ( tf + mu2 * index.totalTermFreq( new Term( field, w ) ) / index.getSumTotalTermFreq( field ) ) / ( len + mu2 );
                collector.put( w, collector.getOrDefault( w, 0.0 ) + pw * dw );
            }
        }
        return Utils.getTop( Utils.norm( collector ), numfbterms );
    }

    public Map<String, Double> estimateQueryModelRM3( List<String> terms, Map<String, Double> rm1, double weight_org ) throws IOException {

        Map<String, Double> mle = new HashMap<>();
        for ( String term : terms ) {
            mle.put( term, mle.getOrDefault( term, 0.0 ) + 1.0 );
        }
        for ( String w : mle.keySet() ) {
            mle.put( w, mle.get( w ) / terms.size() );
        }

        Set<String> v = new TreeSet<>();
        v.addAll( terms );
        v.addAll( rm1.keySet() );

        Map<String, Double> rm3 = new HashMap<>();
        for ( String w : v ) {
            rm3.put( w, weight_org * mle.getOrDefault( w, 0.0 ) + ( 1 - weight_org ) * rm1.getOrDefault( w, 0.0 ) );
        }

        return rm3;
    }

}

