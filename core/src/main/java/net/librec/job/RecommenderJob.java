/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
/*
at bpr : kcv에서 모든 값 올바르게 나옴. avg는 안됨
at myrec : kcv에서 첫값만 똥값 . avg는 안됨
*/

package net.librec.job;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.data.DataModel;
import net.librec.data.DataSplitter;
import net.librec.data.splitter.KCVDataSplitter;
import net.librec.data.splitter.LOOCVDataSplitter;
import net.librec.eval.Measure.MeasureValue;
import net.librec.eval.RecommenderEvaluator;
import net.librec.filter.RecommendedFilter;
import net.librec.math.algorithm.Randoms;
import net.librec.recommender.Recommender;
import net.librec.recommender.RecommenderContext;
import net.librec.recommender.cf.ranking.MyRecommender;
import net.librec.recommender.cf.ranking.MyRecommender2;
import net.librec.recommender.item.RecommendedItem;
import net.librec.similarity.RecommenderSimilarity;
import net.librec.util.DriverClassUtil;
import net.librec.util.FileUtil;
import net.librec.util.JobUtil;
import net.librec.util.ReflectionUtil;

/*
4개의 param에 대해서, top-n 제외하고 3중 for-loop 돌 경우
*/

/**
 * RecommenderJob
 *
 * @author WangYuFeng
 */
public class RecommenderJob {
	/**
	 * LOG
	 */
	protected final Log LOG = LogFactory.getLog(RecommenderJob.class);

	private Configuration conf;

	private DataModel dataModel;

	private Map<Integer, Map<String, List<Double>>> agg_cvEvalResults;

	private Map<String, Double> evalResultList;

	private Map<String, List<String>> paramInfo;

	private String globalOutputPath;

	private double alpha, beta, gamma;

	private int theta;

	public void setTheta(int theta) {
		this.theta = theta;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	public RecommenderJob(Configuration conf) {
		this.conf = conf;
		Long seed = conf.getLong("rec.random.seed");
		if (seed != null) {
			Randoms.seed(seed);
		}
		setJobId(JobUtil.generateNewJobId());
	}

	/**
	 * run Job
	 *
	 * @throws LibrecException
	 *             If an LibrecException error occurs.
	 * @throws ClassNotFoundException
	 *             if can't find the class of filter
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	public void runJob() throws LibrecException, ClassNotFoundException, IOException {
		String modelSplit = conf.get("data.model.splitter");

		paramInfo = new HashMap<String, List<String>>();
		setParamInfo();

		for (int b = 0; b < paramInfo.get("learnrate").size(); b++) {
			for (int c = 0; c < paramInfo.get("iteration").size(); c++) {
				for (int d = 0; d < paramInfo.get("factor").size(); d++) {

					conf.set("rec.iterator.learnrate", paramInfo.get("learnrate").get(b));
					conf.set("rec.iterator.maximum", paramInfo.get("iteration").get(c));
					conf.set("rec.factor.number", paramInfo.get("factor").get(d));

					switch (modelSplit) {
					case "kcv": {
						int cvNumber = conf.getInt("data.splitter.cv.number", 1);
						agg_cvEvalResults = new HashMap<>();

						for (int i = 1; i <= cvNumber; i++) {
							LOG.info("Splitter info: the index of " + modelSplit + " splitter times is " + i);
							conf.set("data.splitter.cv.index", String.valueOf(i));
							executeRecommenderJob();
						}
						printCVAverageResult();
						break;
					}
					case "loocv": {
						String loocvType = conf.get("data.splitter.loocv");
						if (StringUtils.equals("userdate", loocvType) || StringUtils.equals("itemdate", loocvType)) {
							executeRecommenderJob();
						} else {
							// cvEvalResults = new HashMap<>();
							for (int i = 1; i <= conf.getInt("data.splitter.cv.number", 1); i++) {
								LOG.info("Splitter info: the index of " + modelSplit + " splitter times is " + i);
								conf.set("data.splitter.cv.index", String.valueOf(i));
								executeRecommenderJob();
							}
							printCVAverageResult();
						}
						break;
					}
					case "testset": {
						executeRecommenderJob();
						break;
					}
					case "givenn": {
						executeRecommenderJob();
						break;
					}
					case "ratio": {
						executeRecommenderJob();
						break;
					}
					}
				}
			}
		}
	}

	/**
	 * execute Recommender Job
	 *
	 * @throws LibrecException
	 *             If an LibrecException error occurs.
	 * @throws ClassNotFoundException
	 *             if can't find the class of filter
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	@SuppressWarnings("unchecked")
	private void executeRecommenderJob() throws ClassNotFoundException, LibrecException, IOException {
		generateDataModel();
		RecommenderContext context = new RecommenderContext(conf, dataModel);
		generateSimilarity(context);
		Recommender recommender = (Recommender) ReflectionUtil.newInstance((Class<Recommender>) getRecommenderClass(),
				conf);
		if (recommender instanceof MyRecommender) {
			((MyRecommender) recommender).setAlpha(this.alpha);
			((MyRecommender) recommender).setBeta(this.beta);
			((MyRecommender) recommender).setGamma(this.gamma);
		}

		if (recommender instanceof MyRecommender2) {
			((MyRecommender2) recommender).setTheta(this.theta);
		}

		recommender.recommend(context);
		List<RecommendedItem> recommendedList = recommender.getRecommendedList();
		recommendedList = filterResult(recommendedList);

		/* 한번의 실행에 여러개의 topn을 측정(setParamInfo에 저장된 값들로) */

		for (int i = 0; i < paramInfo.get("topN").size(); i++) {
			String newTopN = paramInfo.get("topN").get(i);
			conf.set("rec.recommender.ranking.topn", newTopN);
			recommender.setTopN(Integer.parseInt(newTopN));
			executeEvaluator(recommender);
			saveResult(recommendedList);
		}
	}

	/**
	 * Generate data model.
	 *
	 * @throws ClassNotFoundException
	 * @throws IOException
	 * @throws LibrecException
	 */
	@SuppressWarnings("unchecked")
	private void generateDataModel() throws ClassNotFoundException, IOException, LibrecException {
		if (null == dataModel) {
			dataModel = ReflectionUtil.newInstance((Class<DataModel>) this.getDataModelClass(), conf);
		}
		dataModel.buildDataModel();

	}

	/**
	 * Generate similarity.
	 *
	 * @param context
	 *            recommender context
	 */
	private void generateSimilarity(RecommenderContext context) {
		String[] similarityKeys = conf.getStrings("rec.recommender.similarities");
		if (similarityKeys != null && similarityKeys.length > 0) {
			for (int i = 0; i < similarityKeys.length; i++) {
				if (getSimilarityClass() != null) {
					RecommenderSimilarity similarity = (RecommenderSimilarity) ReflectionUtil
							.newInstance(getSimilarityClass(), conf);
					conf.set("rec.recommender.similarity.key", similarityKeys[i]);
					similarity.buildSimilarityMatrix(dataModel);
					if (i == 0) {
						context.setSimilarity(similarity);
					}
					context.addSimilarities(similarityKeys[i], similarity);
				}
			}
		}
	}

	/**
	 * Filter the results.
	 *
	 * @param recommendedList
	 *            list of recommended items
	 * @return recommended List
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	private List<RecommendedItem> filterResult(List<RecommendedItem> recommendedList)
			throws ClassNotFoundException, IOException {
		if (getFilterClass() != null) {
			RecommendedFilter filter = (RecommendedFilter) ReflectionUtil.newInstance(getFilterClass(), null);
			recommendedList = filter.filter(recommendedList);
		}
		return recommendedList;
	}

	/**
	 * Execute evaluator.
	 *
	 * @param recommender
	 *            recommender algorithm
	 * @throws LibrecException
	 *             if error occurs
	 * @throws IOException
	 *             if I/O error occurs
	 * @throws ClassNotFoundException
	 *             if class not found error occurs
	 */
	private void executeEvaluator(Recommender recommender) throws ClassNotFoundException, IOException, LibrecException {
		evalResultList = new HashMap<String, Double>();

		if (conf.getBoolean("rec.eval.enable")) {
			String[] evalClassKeys = conf.getStrings("rec.eval.classes");
			Map<String, List<Double>> cvEvalResults = new HashMap<>();
			if (evalClassKeys != null && evalClassKeys.length > 0) {// Run the evaluator which is
				// designated.
				for (int classIdx = 0; classIdx < evalClassKeys.length; ++classIdx) {
					RecommenderEvaluator evaluator = (RecommenderEvaluator) ReflectionUtil
							.newInstance(getEvaluatorClass(evalClassKeys[classIdx]), null);
					evaluator.setTopN(conf.getInt("rec.recommender.ranking.topn", 10));
					double evalValue = recommender.evaluate(evaluator);
					LOG.info("Evaluator info:" + evaluator.getClass().getSimpleName() + " is " + evalValue);
					cvEvalResults = collectCVResults(evaluator.getClass().getSimpleName(), evalValue, cvEvalResults);
				}
			} else {// Run all evaluators
				Map<MeasureValue, Double> evalValueMap = recommender.evaluateMap();
				boolean isCV = false;
				if (evalValueMap != null && evalValueMap.size() > 0) {
					int agg_idx = 0;
					for (Map.Entry<MeasureValue, Double> entry : evalValueMap.entrySet()) {
						String evalName = null;
						if (entry != null && entry.getKey() != null) {
							// in case of top-n recommendation
							if (entry.getKey().getTopN() != null && entry.getKey().getTopN() > 0) {
								LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " top "
										+ entry.getKey().getTopN() + " is " + entry.getValue());
								evalName = entry.getKey().getMeasure() + "";
								evalResultList.put(evalName, entry.getValue());
								agg_idx = entry.getKey().getTopN();
							} else { // else
								LOG.info("Evaluator value:" + entry.getKey().getMeasure() + " is " + entry.getValue());
								evalName = entry.getKey().getMeasure() + "";
							}
							if (null != cvEvalResults) {
								isCV = true;
								cvEvalResults = collectCVResults(evalName, entry.getValue(), cvEvalResults);
							}
						}
					}
					if (null != dataModel
							&& (dataModel.getDataSplitter() instanceof KCVDataSplitter
									|| dataModel.getDataSplitter() instanceof LOOCVDataSplitter)
							&& null != conf.getInt("data.splitter.cv.index")) {
						if (agg_cvEvalResults.containsKey(agg_idx)) {
							for (String metric : cvEvalResults.keySet()) {
								agg_cvEvalResults.get(agg_idx).get(metric).addAll(cvEvalResults.get(metric));
							}
						} else {
							agg_cvEvalResults.put(agg_idx, cvEvalResults);
						}
					}
				}
			}
		}
	}

	public String getGlobalOutputPath(String topn) throws ClassNotFoundException, IOException {
		String res = conf.get("dfs.result.dir") + "/" + conf.get("data.input.path") + "-"
				+ DriverClassUtil.getDriverName(getRecommenderClass()) + "-output/" + "[Top-" + topn + "]"
				+ "[Learnrate=" + conf.get("rec.iterator.learnrate") + "]" + "[Iteration="
				+ conf.get("rec.iterator.maximum") + "]" + "[Factor=" + conf.get("rec.factor.number") + "]["
				+ conf.get("data.model.splitter");
		return res;
	}

	/**
	 * Save result.
	 *
	 * @param recommendedList
	 *            list of recommended items
	 * @throws LibrecException
	 *             if error occurs
	 * @throws IOException
	 *             if I/O error occurs
	 * @throws ClassNotFoundException
	 *             if class not found error occurs
	 */
	public void saveResult(List<RecommendedItem> recommendedList)
			throws LibrecException, IOException, ClassNotFoundException {
		// parameters
		String iteratorLearnrate = conf.get("rec.iterator.learnrate");
		String factorNumber = conf.get("rec.factor.number");
		String iteratorMaximum = conf.get("rec.iterator.maximum");
		String rankingTopN = conf.get("rec.recommender.ranking.topn");
		String coeff = "[" + String.valueOf(alpha) + " " + String.valueOf(beta) + " " + String.valueOf(gamma);
		if (recommendedList != null && recommendedList.size() > 0) {
			// make output path
			String algoSimpleName = DriverClassUtil.getDriverName(getRecommenderClass());
			String localOutputPath = null;
			globalOutputPath = conf.get("dfs.result.dir") + "/" + conf.get("data.input.path") + "-" + algoSimpleName
					+ "-output/" + "[Top-" + rankingTopN + "]" + "[Learnrate=" + iteratorLearnrate + "]" + "[Iteration="
					+ iteratorMaximum + "]" + "[Factor=" + factorNumber + "][" + conf.get("data.model.splitter") + "]"
					+ coeff;

			if (null != dataModel
					&& (dataModel.getDataSplitter() instanceof KCVDataSplitter
							|| dataModel.getDataSplitter() instanceof LOOCVDataSplitter)
					&& null != conf.getInt("data.splitter.cv.index")) {
				// if data.model.splitter = KCV || LOOCV
				localOutputPath = globalOutputPath + "-" + String.valueOf(conf.getInt("data.splitter.cv.index")) + "]";
			} else {
				localOutputPath = globalOutputPath + "]";
			}

			localOutputPath += ".txt";
			LOG.info("Result path is " + localOutputPath);
			// convert itemList to string
			StringBuilder sb = new StringBuilder();

			for (String metric : evalResultList.keySet()) {
				Double result = evalResultList.get(metric);
				sb.append(metric).append("\t").append(result).append("\n");
			}
			String resultData = sb.toString();

			// save resultData
			try {
				FileUtil.writeString(localOutputPath, resultData);
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
	}

	/**
	 * Print the average evaluate results when using cross validation.
	 * 
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	private void printCVAverageResult() throws ClassNotFoundException, IOException {

		for (int i = 0; i < agg_cvEvalResults.size(); i++) {
			StringBuilder sb = new StringBuilder();
			String targetN = paramInfo.get("topN").get(i);
			String localOutputPath = getGlobalOutputPath(targetN) + "-avg].txt";

			LOG.info("Average Evaluation Result of Cross Validation:");

			for (Map.Entry<String, List<Double>> entry : agg_cvEvalResults.get(Integer.parseInt(targetN)).entrySet()) {
				String evalName = entry.getKey();
				List<Double> evalList = entry.getValue();
				double sum = 0.0;
				for (double value : evalList) {
					sum += value;
				}
				double avgEvalResult = sum / evalList.size();
				LOG.info("Evaluator value:" + evalName + " is " + avgEvalResult);
				sb.append(evalName).append('\t').append(avgEvalResult).append('\n');
			}
			String resultData = sb.toString();
			try {
				FileUtil.writeString(localOutputPath, resultData);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}

	/**
	 * Collect the evaluate results when using cross validation.
	 *
	 * @param evalName
	 *            name of the evaluator
	 * @param evalValue
	 *            value of the evaluate result
	 */
	private Map<String, List<Double>> collectCVResults(String evalName, Double evalValue,
			Map<String, List<Double>> cvEvalResults) {
		DataSplitter splitter = dataModel.getDataSplitter();
		if (splitter != null && (splitter instanceof KCVDataSplitter || splitter instanceof LOOCVDataSplitter)) {
			if (cvEvalResults.containsKey(evalName)) {
				cvEvalResults.get(evalName).add(evalValue);
			} else {
				List<Double> newList = new ArrayList<>();
				newList.add(evalValue);
				cvEvalResults.put(evalName, newList);
			}
			return cvEvalResults;
		}
		return null;
	}

	private void setJobId(String jobId) {
		conf.set("rec.job.id", jobId);
	}

	public void setRecommenderClass(String jobClass) {
		conf.set("rec.recommender.class", jobClass);
	}

	public void setRecommenderClass(Class<Recommender> jobClass) {
		conf.set("rec.recommender.class", jobClass.getName());
	}

	public void setParamInfo() {
		List<String> topNs = Arrays.asList(new String[] { "5", "10", "20", "30" });
		List<String> learnrates = Arrays.asList(new String[] { "0.01" });
		List<String> iterations = Arrays.asList(new String[] { "100" });
		List<String> factors = Arrays.asList(new String[] { "10" });

		paramInfo.put("topN", topNs);
		paramInfo.put("learnrate", learnrates);
		paramInfo.put("iteration", iterations);
		paramInfo.put("factor", factors);
	}

	/**
	 * Get data model class.
	 *
	 * @return {@code Class<? extends DataModel>} object
	 * @throws ClassNotFoundException
	 *             if the class is not found
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	@SuppressWarnings("unchecked")
	public Class<? extends DataModel> getDataModelClass() throws ClassNotFoundException, IOException {
		return (Class<? extends DataModel>) DriverClassUtil.getClass(conf.get("data.model.format"));
	}

	/**
	 * Get similarity class
	 *
	 * @return similarity class object
	 */
	@SuppressWarnings("unchecked")
	public Class<? extends RecommenderSimilarity> getSimilarityClass() {
		try {
			return (Class<? extends RecommenderSimilarity>) DriverClassUtil.getClass(conf.get("rec.similarity.class"));
		} catch (ClassNotFoundException e) {
			return null;
		}
	}

	/**
	 * Get recommender class. {@code Recommender}.
	 *
	 * @return recommender class object
	 * @throws ClassNotFoundException
	 *             if can't find the class of recommender
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	@SuppressWarnings("unchecked")
	public Class<? extends Recommender> getRecommenderClass() throws ClassNotFoundException, IOException {
		return (Class<? extends Recommender>) DriverClassUtil.getClass(conf.get("rec.recommender.class"));
	}

	/**
	 * Get evaluator class. {@code RecommenderEvaluator}.
	 *
	 * @param evalClassKey
	 *            class key of the evaluator
	 * @return evaluator class object
	 * @throws ClassNotFoundException
	 *             if can't find the class of evaluator
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	@SuppressWarnings("unchecked")
	public Class<? extends RecommenderEvaluator> getEvaluatorClass(String evalClassKey)
			throws ClassNotFoundException, IOException {
		return (Class<? extends RecommenderEvaluator>) DriverClassUtil.getClass(evalClassKey);
	}

	/**
	 * Get filter class. {@code RecommendedFilter}.
	 *
	 * @return evaluator class object
	 * @throws ClassNotFoundException
	 *             if can't find the class of filter
	 * @throws IOException
	 *             If an I/O error occurs.
	 */
	@SuppressWarnings("unchecked")
	public Class<? extends RecommendedFilter> getFilterClass() throws ClassNotFoundException, IOException {
		return (Class<? extends RecommendedFilter>) DriverClassUtil.getClass(conf.get("rec.filter.class"));
	}

}