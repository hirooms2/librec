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
package net.librec.recommender.cf.ranking;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.conf.Configured;
import net.librec.data.convertor.TextDataConvertor;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.recommender.Recommender;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 
 * Triple Wise Learning
 * 
 * @author KIMTAEHO
 *
 */
@ModelData({ "isRanking", "bpr", "userFactors", "itemFactors" })
public class MyRecommender extends MatrixFactorizationRecommender {
	private List<Set<Integer>> INuserItemsSet;
	List<Integer> UNitemList, INitemList;
	List<Integer> INitemSet, UNitemSet;
	int NEItemIdx;
	SparseMatrix interesting;
	private int N;
	double alpha, gamma, beta;
	static final int SETSIZE = 5;

	@Override
	protected void setup() throws LibrecException {

		super.setup();

//		Recommender wrmf = new WRMFRecommender();
//		wrmf.recommend(context);

		String inputDataPath = conf.get(Configured.CONF_DFS_DATA_DIR) + "/interesting";
		TextDataConvertor textDataConvertor = new TextDataConvertor(inputDataPath);
		try {

			textDataConvertor.processData();
			interesting = textDataConvertor.getPreferenceMatrix();

			interesting.setPreferenceList();

			LOG.info("make interest.txt");
			PrintWriter pw = new PrintWriter("interest.txt");
			pw.print(interesting);
			pw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	protected void trainModel() throws LibrecException {
		INuserItemsSet = getUserItemsSet(trainMatrix);

		System.out.println(alpha + " " + beta + " " + gamma);
		try {
			LOG.info("make traing.txt");

			PrintWriter pw = new PrintWriter("traing.txt");
			pw.print(trainMatrix);
			pw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		try {
			LOG.info("make test.txt");

			PrintWriter pw = new PrintWriter("test.txt");
			pw.print(testMatrix);
			pw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int iter = 1; iter <= numIterations; iter++) {

			loss = 0.0d;
			double IA = 0, NA = 0, UA = 0;
			int cnt = 0;
			int cnt2 = 0;
			for (int sampleCount = 0, smax = numUsers * 100; sampleCount < smax; sampleCount++) {

				// randomly draw (userIdx, posItemIdx, negItemIdx)
				int userIdx;
				while (true) {

					userIdx = Randoms.uniform(numUsers);
					INitemList = new ArrayList<>();
					INitemList.addAll(INuserItemsSet.get(userIdx));

					N = INitemList.size();
					if (INitemList.size() == 0 || INitemList.size() == numItems)
						continue;

					UNitemList = interesting.getColumns_UN(userIdx, N);

					do {
						NEItemIdx = Randoms.uniform(numItems);
						if (!INitemList.contains(NEItemIdx) && !UNitemList.contains(NEItemIdx))
							break;
					} while (true);

					INitemSet = new ArrayList<>();
					UNitemSet = new ArrayList<>();

					do {
						INitemSet.add(INitemList.get(Randoms.uniform(INitemList.size())));
					} while (INitemSet.size() < SETSIZE);

					do {
						int v = UNitemList.get(Randoms.uniform(UNitemList.size()));
						if (!INitemList.contains(v)) {
							UNitemSet.add(v);
						} else {
							cnt2++;
						}

					} while (UNitemSet.size() < SETSIZE);

					// for (int i = 0; i < N; i++) {
					// if (interesting.get(userIdx, UNitemList.get(i)) >
					// interesting.get(userIdx, INitemList.get(i))) {
					// System.out.println("FUCKFUCKFUCK!!!!! You are a piece of
					// shit");
					// System.out.println(interesting.get(userIdx,
					// UNitemList.get(i)) + " "
					// + interesting.get(userIdx, INitemList.get(i)));
					// }
					// }

					break;
				}

				double INPredictRatingAvg = 0;
				double NEPredictRatingAvg = 0;
				double UNPredictRatingAvg = 0;

				if (INitemSet.size() != SETSIZE || UNitemSet.size() != SETSIZE)
					System.out.println("FUCK");

				for (Integer idx : INitemSet) {
					INPredictRatingAvg += predict(userIdx, idx);
				}
				INPredictRatingAvg /= SETSIZE;
				IA += INPredictRatingAvg;

				NEPredictRatingAvg += predict(userIdx, NEItemIdx);
				NA += NEPredictRatingAvg;

				for (Integer idx : UNitemSet) {
					UNPredictRatingAvg += predict(userIdx, idx);
				}
				UNPredictRatingAvg /= SETSIZE;
				UA += UNPredictRatingAvg;
				cnt++;

				alpha = 1;
				beta = 1;
				gamma = 0;

				// update parameters
				// double INPredictRating = predict(userIdx, INItemIdx);
				// double NEPredictRating = predict(userIdx, NEItemIdx);
				// double UNPredictRating = predict(userIdx, UNItemIdx);

				double diffValueA = (INPredictRatingAvg - NEPredictRatingAvg) * alpha;
				double diffValueB = (INPredictRatingAvg - UNPredictRatingAvg) * beta;
				double diffValueC = (NEPredictRatingAvg - UNPredictRatingAvg) * gamma;

				double lossValue = -Math.log(Maths.logistic(diffValueA)) - Math.log(Maths.logistic(diffValueB))
						- Math.log(Maths.logistic(diffValueC));
				loss += lossValue;

				double deriValueA = Maths.logistic(-diffValueA);
				double deriValueB = Maths.logistic(-diffValueB);
				double deriValueC = Maths.logistic(-diffValueC);

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);

					double INItemFactorValueAvg = 0;
					double NEItemFactorValueAvg = 0;
					double UNItemFactorValueAvg = 0;

					for (Integer idx : INitemSet) {
						INItemFactorValueAvg += itemFactors.get(idx, factorIdx);
					}
					INItemFactorValueAvg /= SETSIZE;

					NEItemFactorValueAvg = itemFactors.get(NEItemIdx, factorIdx);

					for (Integer idx : UNitemSet) {
						UNItemFactorValueAvg += itemFactors.get(idx, factorIdx);
					}
					UNItemFactorValueAvg /= SETSIZE;

					userFactors.add(userIdx, factorIdx,
							learnRate * (deriValueA * alpha * (INItemFactorValueAvg - NEItemFactorValueAvg)
									+ deriValueB * beta * (INItemFactorValueAvg - UNItemFactorValueAvg)
									+ deriValueC * gamma * (NEItemFactorValueAvg - UNItemFactorValueAvg)
									- regUser * userFactorValue));
					for (Integer idx : INitemSet) {
						itemFactors.add(idx, factorIdx, learnRate * (deriValueA * alpha * userFactorValue / SETSIZE
								+ deriValueB * beta * userFactorValue / SETSIZE - regItem * INItemFactorValueAvg));
					}

					itemFactors.add(NEItemIdx, factorIdx, learnRate * (deriValueA * (-alpha) * userFactorValue
							+ deriValueC * (gamma) * userFactorValue - regItem * NEItemFactorValueAvg));

					for (Integer idx : UNitemSet) {
						itemFactors.add(idx, factorIdx, learnRate * (deriValueB * (-beta) * userFactorValue / SETSIZE
								+ deriValueC * (-gamma) * userFactorValue / SETSIZE - regItem * UNItemFactorValueAvg));
					}

					loss += regUser * userFactorValue * userFactorValue
							+ regItem * INItemFactorValueAvg * INItemFactorValueAvg
							+ regItem * NEItemFactorValueAvg * NEItemFactorValueAvg
							+ regItem * UNItemFactorValueAvg * UNItemFactorValueAvg;
				}

			}
			System.out.println(IA / cnt + " " + NA / cnt + " " + UA / cnt + " " + cnt2);

			if (isConverged(iter) && earlyStop) {
				break;
			}
			updateLRate(iter);
		}

	}

	private List<Set<Integer>> getUserItemsSet(SparseMatrix sparseMatrix) {

		List<Set<Integer>> userItemsSet = new ArrayList<>();
		for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
			userItemsSet.add(new HashSet(sparseMatrix.getColumns(userIdx)));
		}
		return userItemsSet;
	}

	public void setAlpha(double alpha) {
		// TODO Auto-generated method stub
		this.alpha = alpha;
	}

	public void setBeta(double beta) {
		// TODO Auto-generated method stub
		this.beta = beta;
	}

	public void setGamma(double gamma) {
		// TODO Auto-generated method stub
		this.gamma = gamma;
	}
}