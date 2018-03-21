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
import java.util.Scanner;
import java.util.Set;

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
	Set<Integer> UNitemSet;
	Set<Integer> NEitemSet;

	@Override
	protected void setup() throws LibrecException {
		super.setup();
	}

	@Override
	protected void trainModel() throws LibrecException {

		INuserItemsSet = getUserItemsSet_IN(trainMatrix);

		for (int iter = 1; iter <= numIterations; iter++) {

			int shot = 0;
			int noshot = 0;
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

					UNitemSet = getItemsSet_UN(trainMatrix, userIdx, N);
					// NEitemSet = getItemsSet_NE(trainMatrix, userIdx, N);

					List<Integer> INitemList = new ArrayList<>();
					INitemList.addAll(INitemSet);

					// List<Integer> UNitemList = new ArrayList<>();
					// UNitemList.addAll(UNitemSet);

					// List<Integer> NEitemList = new ArrayList<>();
					// NEitemList.addAll(NEitemSet);

					INItemIdx = INitemList.get(Randoms.uniform(N));
					// UNItemIdx = UNitemList.get(Randoms.uniform(N));
					// NEItemIdx = NEitemList.get(Randoms.uniform(N));
					do {
						NEItemIdx = Randoms.uniform(numItems);
						if (!INitemSet.contains(NEItemIdx))
							break;
					} while (true);

					if (INitemSet.contains(NEItemIdx))
						System.out.println("NO1");

					// if (UNitemSet.contains(NEItemIdx))
					// System.out.println("NO2");
					break;

				}

				// update parameters
				double INPredictRating = predict(userIdx, INItemIdx);
				double NEPredictRating = predict(userIdx, NEItemIdx);

				double alpha = 1;
				double k = 20;

				if (UNitemSet.contains(NEItemIdx)) {
					alpha = 20;
				}

				double diffValue = alpha * (INPredictRating - NEPredictRating);
				// double diffValue = (INPredictRating - NEPredictRating)
				// * Maths.logistic(-trainMatrix.get(userIdx, NEItemIdx));
				double lossValue = -Math.log(Maths.logistic(diffValue));
				loss += lossValue;

				double deriValue = Maths.logistic(-diffValue);

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double INItemFactorValue = itemFactors.get(INItemIdx, factorIdx);
					double NEItemFactorValue = itemFactors.get(NEItemIdx, factorIdx);

					userFactors.add(userIdx, factorIdx,
							learnRate * (deriValue * (alpha * (INItemFactorValue - NEItemFactorValue))
									- regUser * userFactorValue));
					itemFactors.add(INItemIdx, factorIdx,
							learnRate * (deriValue * alpha * userFactorValue - regItem * INItemFactorValue));
					itemFactors.add(NEItemIdx, factorIdx,
							learnRate * (deriValue * (-alpha) * userFactorValue - regItem * NEItemFactorValue));

					loss += regUser * userFactorValue * userFactorValue
							+ regItem * INItemFactorValue * INItemFactorValue
							+ regItem * NEItemFactorValue * NEItemFactorValue;
				}

			}
			if (isConverged(iter) && earlyStop) {
				break;
			}
			// System.out.println((1.0 * shot) / (shot + noshot));
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