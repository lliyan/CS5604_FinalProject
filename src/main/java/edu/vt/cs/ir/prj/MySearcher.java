package edu.vt.cs.ir.prj;

import edu.vt.cs.ir.utils.DocLengthReader;
import edu.vt.cs.ir.utils.EvalUtils;
import edu.vt.cs.ir.utils.LuceneUtils;
import edu.vt.cs.ir.utils.SearchResult;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.*;
import java.util.*;

public class MySearcher {

    public static void main( String[] args ) {
        try {

            String pathIndex = "D:\\VT\\IR\\HW3\\index_lucene_robust04_krovetz\\index_lucene_robust04_krovetz"; // change it to your own index path
            Analyzer analyzer = LuceneUtils.getAnalyzer( LuceneUtils.Stemming.Krovetz ); // change the stemming setting accordingly

            String pathQueries = "D:\\VT\\IR\\HW3\\queries_robust04"; // change it to your query file path
            String pathQrels = "D:\\VT\\IR\\HW3\\qrels_robust04"; // change it to your qrels file path
            String pathStopwords = "D:\\VT\\IR\\HW3\\stopwords_inquery"; // change to your stop words path

            String field_docno = "docno";
            String field_search = "content";

            MySearcher searcher = new MySearcher( pathIndex );
            searcher.setStopwords( pathStopwords );

            Map<String, String> queries = EvalUtils.loadQueries( pathQueries );
            Map<String, Set<String>> qrels = EvalUtils.loadQrels( pathQrels );

            int top = 1000;
            double mu = 1000;
            double lamda = 0.5;

            ScoringFunction scoreFunc = new QLDirichletSmoothing( mu );
            ScoringFunction scoreFunc1 = new QLJMSmoothing( lamda );

            double[] p10 = new double[ queries.size() ];
            double[] ap = new double[ queries.size() ];
            double[] jp10 = new double[ queries.size() ];
            double[] jap = new double[ queries.size() ];

            System.out.printf(
                    "%-10s%8s%8s%8s%8s\n",
                    "qid",
                    "p10-QLD",
                    "ap-QLD",
                    "p10-QLJ",
                    "ap-QLJ"
            );

            int ix = 0;
            for ( String qid : queries.keySet() ) {

                String query = queries.get( qid );
                List<String> terms = LuceneUtils.tokenize( query, analyzer );
                String[] termsarray = new String[ terms.size() ];
                for ( int k = 0; k < termsarray.length; k++ ) {
                    termsarray[ k ] = terms.get( k );
                }

                List<SearchResult> results = searcher.search( field_search, terms, scoreFunc, top );
                List<SearchResult> results1 = searcher.search( field_search, terms, scoreFunc1, top );
                SearchResult.dumpDocno( searcher.index, field_docno, results );
                SearchResult.dumpDocno( searcher.index, field_docno, results1 );

                p10[ ix ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                ap[ ix ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );
                jp10[ ix ] = EvalUtils.precision( results1, qrels.get( qid ), 10 );
                jap[ ix ] = EvalUtils.avgPrec( results1, qrels.get( qid ), top );
                ix++;
            }

            System.out.printf(
                    "%-10s%-25s%10.3f%10.3f\n",
                    "QL",
                    "QL",
                    StatUtils.mean( p10 ),
                    StatUtils.mean( ap )
            );
            System.out.printf(
                    "%-10s%-25s%10.3f%10.3f\n",
                    "QL",
                    "QL",
                    StatUtils.mean( jp10 ),
                    StatUtils.mean( jap )
            );


            System.out.println("\n2.2.JM");
            double[] TrainJMp10 = new double[ 150 ];
            double[] TrainJMap = new double[ 150 ];
            double[] TestJMp10 = new double[ 99 ];
            double[] TestJMap = new double[ 99 ];
            Double[] JM = new Double[]{
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            };

            for (int i = 0; i <9; i++) {
                String str = String.format("\nlamda = " + JM[i]);
                System.out.println(str);
                ScoringFunction QLJM = new QLJMSmoothing( JM[i] );
                int x = 0;
                int y = 0;
                for ( String qid : queries.keySet() ) {
                    if(x < 150){
                        String query = queries.get( qid );
                        List<String> terms = LuceneUtils.tokenize( query, analyzer );
                        String[] termsarray = new String[ terms.size() ];
                        for ( int k = 0; k < termsarray.length; k++ ) {
                            termsarray[ k ] = terms.get( k );
                        }

                        List<SearchResult> results = searcher.search( field_search, terms, QLJM, top );
                        SearchResult.dumpDocno( searcher.index, field_docno, results );

                        TrainJMp10[ x ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                        TrainJMap[ x ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );

//                        System.out.printf(
//                                "%-10s%8.3f%8.3f\n",
//                                qid,
//                                TrainJMp10[ x ],
//                                TrainJMap[ x ]
//                        );
                    }
                    else{
                        String query = queries.get( qid );
                        List<String> terms = LuceneUtils.tokenize( query, analyzer );
                        String[] termsarray = new String[ terms.size() ];
                        for ( int k = 0; k < termsarray.length; k++ ) {
                            termsarray[ k ] = terms.get( k );
                        }

                        List<SearchResult> results = searcher.search( field_search, terms, QLJM, top );
                        SearchResult.dumpDocno( searcher.index, field_docno, results );

                        TestJMp10[ y ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                        TestJMap[ y ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );

//                        System.out.printf(
//                                "%-10s%8.3f%8.3f\n",
//                                qid,
//                                TestJMp10[ y ],
//                                TestJMap[ y ]
//                        );
                        y++;
                    }

                    x++;

                }
                System.out.printf(
                        "%-10s%10.3f%10.3f\n",
                        "QL Training",
                        StatUtils.mean( TrainJMp10 ),
                        StatUtils.mean( TrainJMap )
                );
                System.out.printf(
                        "%-10s%10.3f%10.3f\n",
                        "QL Testing",
                        StatUtils.mean( TestJMp10 ),
                        StatUtils.mean( TestJMap )
                );
            }

            System.out.println("\n2.2.Dirichlet");
            double[] TrainDp10 = new double[ 150 ];
            double[] TrainDap = new double[ 150 ];
            double[] TestDp10 = new double[ 99 ];
            double[] TestDap = new double[ 99 ];
            Double[] Dirichlet = new Double[]{
                    500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0
            };

            for (int i = 0; i <Dirichlet.length; i++) {
                String str = String.format("\nmu = " + Dirichlet[i]);
                System.out.println(str);
                ScoringFunction QLD = new QLDirichletSmoothing(Dirichlet[i]);
                int x = 0;
                int y = 0;
                for ( String qid : queries.keySet() ) {
                    if(x < 150){
                        String query = queries.get( qid );
                        List<String> terms = LuceneUtils.tokenize( query, analyzer );
                        String[] termsarray = new String[ terms.size() ];
                        for ( int k = 0; k < termsarray.length; k++ ) {
                            termsarray[ k ] = terms.get( k );
                        }

                        List<SearchResult> results = searcher.search( field_search, terms, QLD, top );
                        SearchResult.dumpDocno( searcher.index, field_docno, results );

                        TrainDp10[ x ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                        TrainDap[ x ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );

//                        System.out.printf(
//                                "%-10s%8.3f%8.3f\n",
//                                qid,
//                                TrainJMp10[ x ],
//                                TrainJMap[ x ]
//                        );
                    }
                    else{
                        String query = queries.get( qid );
                        List<String> terms = LuceneUtils.tokenize( query, analyzer );
                        String[] termsarray = new String[ terms.size() ];
                        for ( int k = 0; k < termsarray.length; k++ ) {
                            termsarray[ k ] = terms.get( k );
                        }

                        List<SearchResult> results = searcher.search( field_search, terms, QLD, top );
                        SearchResult.dumpDocno( searcher.index, field_docno, results );

                        TestDp10[ y ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                        TestDap[ y ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );
                        y++;
//                        System.out.printf(
//                                "%-10s%8.3f%8.3f\n",
//                                qid,
//                                TestJMp10[ x ],
//                                TestJMap[ x ]
//                        );
                    }

                    x++;
                }
                System.out.printf(
                        "%-10s%10.3f%10.3f\n",
                        "QL Training",
                        StatUtils.mean( TrainDp10 ),
                        StatUtils.mean( TrainDap )
                );
                System.out.printf(
                        "%-10s%10.3f%10.3f\n",
                        "QL Testing",
                        StatUtils.mean( TestDp10 ),
                        StatUtils.mean( TestDap )
                );
            }


            searcher.close();

        } catch ( Exception e ) {
            e.printStackTrace();
        }
    }

    protected File dirBase;
    protected Directory dirLucene;
    protected IndexReader index;
    protected Map<String, DocLengthReader> doclens;

    protected HashSet<String> stopwords;

    public MySearcher( String dirPath ) throws IOException {
        this( new File( dirPath ) );
    }

    public MySearcher( File dirBase ) throws IOException {
        this.dirBase = dirBase;
        this.dirLucene = FSDirectory.open( this.dirBase.toPath() );
        this.index = DirectoryReader.open( dirLucene );
        this.doclens = new HashMap<>();
        this.stopwords = new HashSet<>();
//        System.out.println(logpdc("content","FBIS4-41991"));
//        System.out.println(logpdc("content","FBIS4-67701"));
//        System.out.println(logpdc("content","FT921-7107"));
//        System.out.println(logpdc("content","FR940617-0-00103"));
//        System.out.println(logpdc("content","FR941212-0-00060"));
//        System.out.println(logpdc("content","FBIS3-25118"));
    }

    public double logpdc( String field, String docno ) throws IOException {
        // implement your Log P(D|Corpus) here

        // |Corpus|
        double corpusLength = index.getSumTotalTermFreq( field );

        Terms vector = index.getTermVector( LuceneUtils.findByDocno( index, "docno", docno ), field );
        TermsEnum terms = vector.iterator();

        BytesRef term;

        double result = 0.0;

        while ( ( term = terms.next() ) != null ) {

            long freq = terms.totalTermFreq();
            long wC = index.totalTermFreq( new Term( field, term.utf8ToString() ));
            double pwc = wC / corpusLength;
            result += (freq*Math.log(pwc));
        }

        return result;


    }

    public void setStopwords( Collection<String> stopwords ) {
        this.stopwords.addAll( stopwords );
    }

    public void setStopwords( String stopwordsPath ) throws IOException {
        setStopwords( new File( stopwordsPath ) );
    }

    public void setStopwords( File stopwordsFile ) throws IOException {
        BufferedReader reader = new BufferedReader( new InputStreamReader( new FileInputStream( stopwordsFile ), "UTF-8" ) );
        String line;
        while ( ( line = reader.readLine() ) != null ) {
            line = line.trim();
            if ( line.length() > 0 ) {
                this.stopwords.add( line );
            }
        }
        reader.close();
    }

    public List<SearchResult> search( String field, List<String> terms, ScoringFunction scoreFunc, int top ) throws IOException {

        Map<String, Double> qfreqs = new TreeMap<>();
        String[] termsarray = new String[ terms.size() ];

        for ( int k = 0; k < termsarray.length; k++ ) {
            termsarray[ k ] = terms.get( k );
        }
        for ( String term : terms ) {
            if ( !stopwords.contains( term ) ) {
                qfreqs.put( term, qfreqs.getOrDefault( term, 0.0 ) + 1 );
            }
        }

        List<PostingsEnum> postings = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        List<Double> tfcs = new ArrayList<>();
        for ( String term : qfreqs.keySet() ) {
            PostingsEnum list = MultiFields.getTermDocsEnum( index, field, new BytesRef( term ), PostingsEnum.FREQS );
            if ( list.nextDoc() != PostingsEnum.NO_MORE_DOCS ) {
                postings.add( list );
                weights.add( qfreqs.get( term ) / terms.size() );
                tfcs.add( 1.0 * index.totalTermFreq( new Term( field, term ) ) );
            }
        }
        return search( postings, weights, tfcs, getDocLengthReader( field ), index.getSumTotalTermFreq( field ), top, termsarray );
    }

    private List<SearchResult> search( List<PostingsEnum> postings, List<Double> weights, List<Double> tfcs, DocLengthReader doclen, double cl, int top, String[] termsarray ) throws IOException {

        PriorityQueue<SearchResult> topResults = new PriorityQueue<>( Comparator.comparingDouble( SearchResult::getScore ).thenComparingInt( SearchResult::getDocid ) );

        List<Double> tfs = new ArrayList<>( weights.size() );
        for ( int ix = 0; ix < weights.size(); ix++ ) {
            tfs.add( 0.0 );
        }
        while ( true ) {

            int docid = Integer.MAX_VALUE;
            for ( PostingsEnum posting : postings ) {
                if ( posting.docID() != PostingsEnum.NO_MORE_DOCS && posting.docID() < docid ) {
                    docid = posting.docID();
                }
            }

            if ( docid == Integer.MAX_VALUE ) {
                break;
            }

            int ix = 0;
            for ( PostingsEnum posting : postings ) {
                if ( docid == posting.docID() ) {
                    tfs.set( ix, 1.0 * posting.freq() );
                    posting.nextDoc();
                } else {
                    tfs.set( ix, 0.0 );
                }
                ix++;
            }
            String docn = LuceneUtils.getDocno( index, "docno", docid );
            double score = modelPRM1(docn, termsarray, 0.1, cl, 5.0, doclen.getLength( docid ));
//                    scoreFunc.score( weights, tfs, tfcs, doclen.getLength( docid ), cl );

            if ( topResults.size() < top ) {
                topResults.add( new SearchResult( docid, null, score ) );
            } else {
                SearchResult result = topResults.peek();
                if ( score > result.getScore() ) {
                    topResults.poll();
                    topResults.add( new SearchResult( docid, null, score ) );
                }
            }
        }

        List<SearchResult> results = new ArrayList<>( topResults.size() );
        results.addAll( topResults );
        Collections.sort( results, ( o1, o2 ) -> o2.getScore().compareTo( o1.getScore() ) );
        return results;
    }

    public double modelPRM1 (String docn, String[] termsarray, double lambda, double cl, double sigma, double dl) throws IOException {


        HashMap<String, HashMap<String,ArrayList<Integer>>> map = new HashMap<>();
        HashMap<String, ArrayList<Integer>> mapNew = new HashMap<>();

        Set<String> fieldset = new HashSet<>();
        fieldset.add("docno");
        HashMap<String, ArrayList<Integer>> temp = new HashMap<>();
        ArrayList<Integer> arrNew = new ArrayList<>();
        for (int k = 0; k < termsarray.length; k++) {
            PostingsEnum posting = MultiFields.getTermDocsEnum(index, "content", new BytesRef(termsarray[k]), PostingsEnum.POSITIONS);
            ArrayList<Integer> arr = new ArrayList<>();
            if (posting != null) {
                int docid;
                while ((docid = posting.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
                    String docno = index.document(docid, fieldset).get("docno");
                    if (docno.equals(docn)) {
                        int freq = posting.freq();
                        for (int j = 0; j < freq; j++) {
                            arr.add(posting.nextPosition());
                            arrNew.add(posting.nextPosition());
                        }
                    }
                }
            }
            temp.put(termsarray[k], arr);
        }
        map.put(docn, temp);
        mapNew.put(docn, arrNew);

        double pwq = 0.0;
        HashMap<String, ArrayList<Integer>> temparr = map.get(docn);

        for (int pos = 0; pos < dl; pos++){
            if (mapNew.get(docn).contains(pos)){
                //c'(w,i)
                double cwi = 0.0;
                for (Integer jpos : mapNew.get(docn)){
                    cwi += 1.0 * Math.exp((-Math.pow((pos-jpos),2.0))/(2.0*Math.pow(sigma,2.0)));
                }
                //P(w|D,i)
                double pwdi = cwi/(Math.sqrt(2.0*Math.PI*Math.pow(sigma,2.0)));
                //P(Q|D,i)
                double pqdi = 1.0;
                for (String term : temparr.keySet()){
                    double tfcs = 1.0 * index.totalTermFreq( new Term( "content", term) );
                    if (temp.get(term).contains(pos)){
                        pqdi *= (1.0-lambda)*(pwdi) + lambda * (tfcs/cl);
                    }else {
                        pqdi *= lambda * (tfcs/cl);
                    }
                }
                pwq += (pqdi/dl);
            }
        }
//        System.out.println(pwq);
        return pwq;
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

    public interface ScoringFunction {

        /**
         * @param weights Weight of the query terms, e.g., P(t|q) or c(t,q).
         * @param tfs     The frequencies of the query terms in documents.
         * @param tfcs    The frequencies of the query terms in the corpus.
         * @param dl      The length of the document.
         * @param cl      The length of the whole corpus.
         * @return
         */
        double score(List<Double> weights, List<Double> tfs, List<Double> tfcs, double dl, double cl);
    }

    public static class QLJMSmoothing implements ScoringFunction {

        protected double lambda;

        public QLJMSmoothing( double lambda ) {
            this.lambda = lambda;
        }

        public double score( List<Double> weights, List<Double> tfs, List<Double> tfcs, double dl, double cl ) {
            // implement your Jelinek-Mercer smoothing
            double score = 1.0;
            for (int i = 0; i < weights.size(); i++ ) {
                double pwc = tfcs.get(i) / cl;
                pwc *= lambda;
                double cwd = tfs.get(i)/dl;
                cwd *= (1-lambda);

                score *= (cwd + pwc);
            }
            return score;
        }
    }

    public static class QLDirichletSmoothing implements ScoringFunction {

        protected double mu;

        public QLDirichletSmoothing( double mu ) {
            this.mu = mu;
        }

        public double score( List<Double> weights, List<Double> tfs, List<Double> tfcs, double dl, double cl ) {

            double score = 1.0;
            for (int i = 0; i < weights.size(); i++ ){
                double pwc = tfcs.get(i)/cl;
                pwc *= mu;
                pwc += tfs.get(i);
                score *= (pwc/(dl + mu));
                //score += (temp * weights.get(i));
            }


            return score;
        }
    }

}
