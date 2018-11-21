package edu.vt.cs.ir.prj;

import edu.vt.cs.ir.search.LuceneTermPostingList;
import edu.vt.cs.ir.search.PostingList;
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

import java.io.File;
import java.io.IOException;
import java.util.*;

public class PRM1 {

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
            double sigma = 0.0;

            MySearcher.ScoringFunction scoreFunc = new MySearcher.QLDirichletSmoothing( mu );

            double cl = searcher.index.getSumTotalTermFreq( "content" );

            double[] p10 = new double[ queries.size() ];
            double[] ap = new double[ queries.size() ];
            int ix = 0;
            for ( String qid : queries.keySet() ) {

                String query = queries.get( qid );
                List<String> terms = LuceneUtils.tokenize( query, analyzer );
                String[] termsarray = new String[ terms.size() ];

                for ( int k = 0; k < termsarray.length; k++ ) {
                    termsarray[ k ] = terms.get( k );
                }

                List<SearchResult> results = searcher.search( field_search, terms, scoreFunc, top );
                SearchResult.dumpDocno( searcher.index, field_docno, results );

                p10[ ix ] = EvalUtils.precision( results, qrels.get( qid ), 10 );
                ap[ ix ] = EvalUtils.avgPrec( results, qrels.get( qid ), top );

                System.out.printf(
                        "%-10s%8.3f%8.3f\n",
                        qid,
                        p10[ ix ],
                        ap[ ix ]
                );
                ix++;

//                System.out.println(qid);
//
//                int docR = 0;
//                ArrayList<String> relArr = new ArrayList<>();
//
//                for (int i = 0; i < results.size(); i++) {
//                    for (String docn : qrels.get(qid)) {
//                        if (docn.equals(results.get(i).getDocno())) {
//                            relArr.add(docn);
//                            docR++;
//                        }
//                    }
//                    if (docR == 10) {
//                        break;
//                    }
//                }
//
//                System.out.println(relArr.size());
//
//                Map<String, Double> prm1 = modelPRM1(qid, relArr, termsarray, searcher, 0.1, cl, sigma);
//                break;


            }

            System.out.printf(
                    "%-10s%-25s%10.3f%10.3f\n",
                    "QL",
                    "QL",
                    StatUtils.mean( p10 ),
                    StatUtils.mean( ap )
            );

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Map<String, Double> modelPRM1 (String qid, ArrayList<String> relArr, String[] termsarray, MySearcher searcher, double lambda, double cl, double sigma) throws IOException {

        System.out.println(relArr);
        //doc length, position,

        HashMap<String, HashMap<String,ArrayList<Integer>>> map = new HashMap<>();
        HashMap<String, ArrayList<Integer>> mapNew = new HashMap<>();
        HashMap<String, Integer> mapDl = new HashMap<>();

        for (int i = 0; i < relArr.size(); i++) {
            int dl = searcher.getDocLengthReader("content").getLength(LuceneUtils.findByDocno(searcher.index, "docno", relArr.get(i)));
            mapDl.put(relArr.get(i), dl);
//            System.out.println(relArr.get(i) + ": " + dl);
            Set<String> fieldset = new HashSet<>();
            fieldset.add("docno");
            HashMap<String, ArrayList<Integer>> temp = new HashMap<>();
            ArrayList<Integer> arrNew = new ArrayList<>();
            for (int k = 0; k < termsarray.length; k++) {
                PostingsEnum posting = MultiFields.getTermDocsEnum(searcher.index, "content", new BytesRef(termsarray[k]), PostingsEnum.POSITIONS);
                ArrayList<Integer> arr = new ArrayList<>();
                if (posting != null) {
                    int docid;
                    while ((docid = posting.nextDoc()) != PostingsEnum.NO_MORE_DOCS) {
                        String docno = searcher.index.document(docid, fieldset).get("docno");
                        if (docno.equals(relArr.get(i))) {
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
            map.put(relArr.get(i), temp);
            mapNew.put(relArr.get(i), arrNew);
        }
//        System.out.println(map);
//        System.out.println(mapDl);
        System.out.println(mapNew);


        double pwq = 0.0;
        for(int i = 0; i < relArr.size(); i++) {
            HashMap<String, ArrayList<Integer>> temp = map.get(relArr.get(i));

            for (int pos = 0; pos < mapDl.get(relArr.get(i)); pos++){
                if (mapNew.get(relArr.get(i)).contains(pos)){
                    //c'(w,i)
                    double cwi = 0.0;
                    for (Integer jpos : mapNew.get(relArr.get(i))){
                        cwi += 1.0 * Math.exp((-Math.pow((pos-jpos),2.0))/(2.0*Math.pow(sigma,2.0)));
                    }

                    //P(w|D,i)
                    double pwdi = cwi/(Math.sqrt(2.0*Math.PI*Math.pow(sigma,2.0)));

                    //P(Q|D,i)
                    double pqdi = 1.0;
                    for (String term : temp.keySet()){
                        double tfcs = 1.0 * searcher.index.totalTermFreq( new Term( "content", term) );
                        if (temp.get(term).contains(pos)){
                            pqdi *= (1.0-lambda)*(pwdi) + lambda * (tfcs/cl);
                        }else {
                            pqdi *= lambda * (tfcs/cl);
                        }
                    }

                    pwq += (pqdi/mapDl.get(relArr.get(i)));
                }
            }
            System.out.println(pwq);


            break;
        }




        return null;
    }
}
