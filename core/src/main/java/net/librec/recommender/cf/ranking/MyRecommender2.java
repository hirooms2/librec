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
IN 미리 뽑고
NE,UN 중 2개중 하나에 속한 아이템을 선택 + 그 선호도의 차이를 학습
(NE = S -{IN U UN})
*/

package net.librec.recommender.cf.ranking;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import net.librec.annotation.ModelData;
import net.librec.common.LibrecException;
import net.librec.data.model.Pair;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

/**
 * Rendle et al., <strong>BPR: Bayesian Personalized Ranking from Implicit
 * Feedback</strong>, UAI 2009.
 *
 * @author GuoGuibing and Keqiang Wang
 */
@ModelData({ "isRanking", "bpr", "userFactors", "itemFactors" })
public class MyRecommender2 extends MatrixFactorizationRecommender {
	private List<Set<Pair>> userItemsSet;
	private List<Set<Integer>> INuserItemsSet;
	private List<Set<Integer>> NEuserItemsSet;
	private List<Set<Integer>> UNuserItemsSet;
	private int N;
	private double theta;

	@Override
	protected void setup() throws LibrecException {
		super.setup();
	}

	public void setTheta(int theta) {
		this.theta = theta;
	}

	@Override
	protected void trainModel() throws LibrecException {
		INuserItemsSet = getUserItemsSet_IN(trainMatrix);
		System.out.println(theta);

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

					Set<Integer> UNitemSet = getItemsSet_UN(trainMatrix, userIdx, N);

					List<Integer> INitemList = new ArrayList<>();
					List<Integer> UNitemList = new ArrayList<>();

					INitemList.addAll(INitemSet);
					UNitemList.addAll(UNitemSet);

					INItemIdx = INitemList.get(Randoms.uniform(N));
					UNItemIdx = UNitemList.get(Randoms.uniform(N));

					Set<Integer> NEitemSet = getItemsSet_NE(trainMatrix, userIdx, N);
					List<Integer> NEitemList = new ArrayList<>();
					NEitemList.addAll(NEitemSet);
					do {
						NEItemIdx = NEitemList.get(Randoms.uniform(N));
					} while (INitemSet.contains(NEItemIdx));

					// do {
					// NEItemIdx = Randoms.uniform(numItems);
					// if (!INitemSet.contains(NEItemIdx))
					// break;
					// } while (true);

					break;

				}
				// System.out.print(N + ": ");
				// System.out.print(trainMatrix.get(userIdx, INItemIdx) + " ");
				// System.out.print(trainMatrix.get(userIdx, NEItemIdx) + " ");
				// System.out.println(trainMatrix.get(userIdx, UNItemIdx));

				// update parameters
				double INPredictRating = predict(userIdx, INItemIdx);
				double UNPredictRating = predict(userIdx, UNItemIdx);
				double NEPredictRating = predict(userIdx, NEItemIdx);
				double diffValue = 0;

				int choose = Randoms.uniform(100);

				// theta = 50;

				// System.out.println(theta);
				if (choose < theta) {
					diffValue = INPredictRating - NEPredictRating;
				} else {
					diffValue = INPredictRating - UNPredictRating;
				}

				double lossValue = -Math.log(Maths.logistic(diffValue));
				loss += lossValue;

				double deriValue = Maths.logistic(-diffValue);

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double INItemFactorValue = itemFactors.get(INItemIdx, factorIdx);
					double UNItemFactorValue = itemFactors.get(UNItemIdx, factorIdx);
					double NEItemFactorValue = itemFactors.get(NEItemIdx, factorIdx);

					if (choose < theta) {
						userFactors.add(userIdx, factorIdx, learnRate
								* (deriValue * (INItemFactorValue - NEItemFactorValue) - regUser * userFactorValue));
						itemFactors.add(INItemIdx, factorIdx,
								learnRate * (deriValue * userFactorValue - regItem * INItemFactorValue));
						itemFactors.add(NEItemIdx, factorIdx,
								learnRate * (deriValue * (-userFactorValue) - regItem * NEItemFactorValue));

						loss += regUser * userFactorValue * userFactorValue
								+ regItem * INItemFactorValue * INItemFactorValue
								+ regItem * NEItemFactorValue * NEItemFactorValue;
					} else {
						userFactors.add(userIdx, factorIdx, learnRate
								* (deriValue * (INItemFactorValue - UNItemFactorValue) - regUser * userFactorValue));
						itemFactors.add(INItemIdx, factorIdx,
								learnRate * (deriValue * userFactorValue - regItem * INItemFactorValue));
						itemFactors.add(UNItemIdx, factorIdx,
								learnRate * (deriValue * (-userFactorValue) - regItem * UNItemFactorValue));

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

	private List<Set<Integer>> getUserItemsSet_IN(SparseMatrix sparseMatrix) {
		sparseMatrix.setPreferenceList();
		List<Set<Integer>> userItemsSet_IN = new ArrayList<>();
		for (int userIdx = 0; userIdx < numUsers; ++userIdx) {
			userItemsSet_IN.add(new HashSet(sparseMatrix.getColumns_IN(userIdx)));
		}
		return userItemsSet_IN;
	}

	private Set<Integer> getItemsSet_UN(SparseMatrix sparseMatrix, int userIdx, int N) {

		return new HashSet(sparseMatrix.getColumns_UN(userIdx, N));
	}

	private Set<Integer> getItemsSet_NE(SparseMatrix sparseMatrix, int userIdx, int N) {

		return new HashSet(sparseMatrix.getColumns_NE(userIdx, N));
	}
}