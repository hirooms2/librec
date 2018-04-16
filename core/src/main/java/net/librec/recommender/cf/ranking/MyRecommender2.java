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
public class MyRecommender2 extends MatrixFactorizationRecommender {
	private List<Set<Integer>> INuserItemsSet;
	SparseMatrix interesting;
	private int N;
	double alpha;

	@Override
	protected void setup() throws LibrecException {

		super.setup();
		String inputDataPath = conf.get(Configured.CONF_DFS_DATA_DIR) + "/interesting";
		TextDataConvertor textDataConvertor = new TextDataConvertor(inputDataPath);
		try {
			textDataConvertor.processData();
			interesting = textDataConvertor.getPreferenceMatrix();
			interesting.setPreferenceList();

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

		System.out.println(alpha);
		try {

			PrintWriter pw = new PrintWriter("traing.txt");
			pw.print(trainMatrix);
			pw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		try {

			PrintWriter pw = new PrintWriter("test.txt");
			pw.print(testMatrix);
			pw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int iter = 1; iter <= numIterations; iter++) {

			loss = 0.0d;
			for (int sampleCount = 0, smax = numUsers * 100; sampleCount < smax; sampleCount++) {

				// randomly draw (userIdx, posItemIdx, negItemIdx)
				int userIdx, INItemIdx, UNItemIdx, NEItemIdx;
				while (true) {
					userIdx = Randoms.uniform(numUsers);
					Set<Integer> INitemSet = INuserItemsSet.get(userIdx);
					N = INitemSet.size();
					if (INitemSet.size() == 0 || INitemSet.size() == numItems)
						continue;

					List<Integer> UNitemList = interesting.getColumns_UN(userIdx, N);
					UNItemIdx = UNitemList.get(Randoms.uniform(N));

					List<Integer> INitemList = new ArrayList<>();
					INitemList.addAll(INitemSet);
					INItemIdx = INitemList.get(Randoms.uniform(N));
					/*
					 * / |IN|=|NE|=|UN|
					 */
					// NEitemSet = getItemsSet_NE(interesting, userIdx, N);
					// List<Integer> NEitemList = new ArrayList<>();
					// NEitemList.addAll(NEitemSet);
					// NEItemIdx = NEitemList.get(Randoms.uniform(N));

					// /*
					// * |IN|=|UN|, NE = S\(IN+UN)
					// */
					// do {
					// NEItemIdx = Randoms.uniform(numItems);
					// if (!INitemSet.contains(NEItemIdx) &&
					// !UNitemSet.contains(NEItemIdx))
					// break;
					// } while (true);

					/*
					 * |IN|=|UN|, NE = S\(IN)
					 */
					do {
						NEItemIdx = Randoms.uniform(numItems);
						if (!INitemSet.contains(NEItemIdx) && !UNitemList.contains(NEItemIdx))
							break;
					} while (true);

					break;

				}

				// update parameters
				double INPredictRating = predict(userIdx, INItemIdx);
				double NEPredictRating = predict(userIdx, NEItemIdx);
				double UNPredictRating = predict(userIdx, UNItemIdx);
				double diffValue;
				double lossValue;

				double choose = Randoms.uniform(100);

				if (choose < alpha) {
					diffValue = (INPredictRating - NEPredictRating);
				} else {
					diffValue = (INPredictRating - UNPredictRating);
				}

				lossValue = -Math.log(Maths.logistic(diffValue));

				loss += lossValue;

				double deriValue = Maths.logistic(-diffValue);
				;

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double INItemFactorValue = itemFactors.get(INItemIdx, factorIdx);
					double NEItemFactorValue = itemFactors.get(NEItemIdx, factorIdx);
					double UNItemFactorValue = itemFactors.get(UNItemIdx, factorIdx);

					if (choose < alpha) {
						userFactors.add(userIdx, factorIdx, learnRate
								* (deriValue * (INItemFactorValue - NEItemFactorValue) - regUser * userFactorValue));
						itemFactors.add(INItemIdx, factorIdx,
								learnRate * (deriValue * userFactorValue - regItem * INItemFactorValue));
						itemFactors.add(NEItemIdx, factorIdx,
								learnRate * (deriValue * -userFactorValue - regItem * NEItemFactorValue));
						loss += regUser * userFactorValue * userFactorValue
								+ regItem * INItemFactorValue * INItemFactorValue
								+ regItem * NEItemFactorValue * NEItemFactorValue;
					} else {
						userFactors.add(userIdx, factorIdx, learnRate
								* (deriValue * (INItemFactorValue - UNItemFactorValue) - regUser * userFactorValue));
						itemFactors.add(INItemIdx, factorIdx,
								learnRate * (deriValue * userFactorValue - regItem * INItemFactorValue));
						itemFactors.add(NEItemIdx, factorIdx,
								learnRate * (deriValue * -userFactorValue - regItem * UNItemFactorValue));
						loss += regUser * userFactorValue * userFactorValue
								+ regItem * INItemFactorValue * INItemFactorValue
								+ regItem * UNItemFactorValue * UNItemFactorValue;
					}
				}

			}
			if (isConverged(iter) && earlyStop) {
				break;
			}
			updateLRate(iter);
		}
	}

	private List<Set<Integer>> getUserItemsSet(SparseMatrix sparseMatrix) {

		List<Set<Integer>> userItemsSet = new ArrayList<>();
		for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
			userItemsSet.add(new HashSet(sparseMatrix.getColumns_IN(userIdx)));
		}
		return userItemsSet;
	}

	public void setAlpha(double alpha) {
		// TODO Auto-generated method stub
		this.alpha = alpha;
	}
}