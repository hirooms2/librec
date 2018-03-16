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
import net.librec.data.model.Pair;
import net.librec.math.algorithm.Maths;
import net.librec.math.algorithm.Randoms;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Rendle et al., <strong>BPR: Bayesian Personalized Ranking from Implicit
 * Feedback</strong>, UAI 2009.
 *
 * @author GuoGuibing and Keqiang Wang
 */
@ModelData({ "isRanking", "bpr", "userFactors", "itemFactors" })
public class MyRecommender extends MatrixFactorizationRecommender {
	private List<Set<Pair>> userItemsSet;
	private List<Set<Integer>> INuserItemsSet;
	private List<Set<Integer>> NEuserItemsSet;
	private List<Set<Integer>> UNuserItemsSet;
	private int N;

	@Override
	protected void setup() throws LibrecException {
		super.setup();
	}

	@Override
	protected void trainModel() throws LibrecException {

		INuserItemsSet = getUserItemsSet_IN(trainMatrix);

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
					Set<Integer> NEitemSet = getItemsSet_NE(trainMatrix, userIdx, N);

					List<Integer> INitemList = new ArrayList<>();
					INitemList.addAll(INitemSet);

					INItemIdx = (INitemList).get(Randoms.uniform(N));

					do {
						UNItemIdx = Randoms.uniform(numItems);
					} while (!UNitemSet.contains(UNItemIdx));

					do {
						NEItemIdx = Randoms.uniform(numItems);
					} while (!NEitemSet.contains(NEItemIdx));

					break;

				}

				// update parameters
				double INPredictRating = predict(userIdx, INItemIdx);
				double UNPredictRating = predict(userIdx, UNItemIdx);
				double NEPredictRating = predict(userIdx, NEItemIdx);

				double f = INPredictRating - NEPredictRating;
				double g = NEPredictRating - UNPredictRating;
				double diffValue = f * g;
				double lossValue = -Math.log(Maths.logistic(diffValue));
				loss += lossValue;

				double deriValue = Maths.logistic(-diffValue);

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double INItemFactorValue = itemFactors.get(INItemIdx, factorIdx);
					double UNItemFactorValue = itemFactors.get(UNItemIdx, factorIdx);
					double NEItemFactorValue = itemFactors.get(NEItemIdx, factorIdx);

					userFactors.add(userIdx, factorIdx,
							learnRate * (deriValue
									* ((INItemFactorValue - NEItemFactorValue) * g
											+ f * (NEItemFactorValue - UNItemFactorValue))
									- regUser * userFactorValue));
					itemFactors.add(INItemIdx, factorIdx,
							learnRate * (deriValue * userFactorValue * g - regItem * INItemFactorValue));
					itemFactors.add(NEItemIdx, factorIdx,
							learnRate * (deriValue * userFactorValue * (f - g) - regItem * NEItemFactorValue));
					itemFactors.add(UNItemIdx, factorIdx,
							learnRate * (deriValue * (-userFactorValue) * f - regItem * UNItemFactorValue));

					loss += regUser * userFactorValue * userFactorValue
							+ regItem * INItemFactorValue * INItemFactorValue
							+ regItem * NEItemFactorValue * NEItemFactorValue
							+ regItem * UNItemFactorValue * UNItemFactorValue;
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

		return new HashSet(sparseMatrix.getColumns_NE(userIdx, N, numUsers));
	}
}