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
import net.librec.math.structure.DenseVector;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
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
	private List<Set<Pair>> userItemsSet;
	private List<Set<Integer>> INuserItemsSet;
	private List<Set<Integer>> NEuserItemsSet;
	private List<Set<Integer>> UNuserItemsSet;
	private int N;
	Set<Integer> UNitemSet;
	Set<Integer> NEitemSet;
	double alpha, gamma, beta;

	private DenseVector itemBiases;

	@Override
	protected void setup() throws LibrecException {

		super.setup();

		itemBiases = new DenseVector(numItems);
		itemBiases.init();
	}

	@Override
	protected void trainModel() throws LibrecException {

		INuserItemsSet = getUserItemsSet_IN(trainMatrix);
		System.out.println(alpha + " " + beta + " " + gamma);

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

					UNitemSet = getItemsSet_UN(trainMatrix, userIdx, N);
					List<Integer> UNitemList = new ArrayList<>();
					UNitemList.addAll(UNitemSet);
					UNItemIdx = UNitemList.get(Randoms.uniform(N));

					List<Integer> INitemList = new ArrayList<>();
					INitemList.addAll(INitemSet);
					INItemIdx = INitemList.get(Randoms.uniform(N));

					/*
					 * / |IN|=|NE|=|UN|
					 */
					// NEitemSet = getItemsSet_NE(trainMatrix, userIdx, N);
					// List<Integer> NEitemList = new ArrayList<>();
					// NEitemList.addAll(NEitemSet);
					// NEItemIdx = NEitemList.get(Randoms.uniform(N));

					// /*
					// * |IN|=|UN|, NE = S\(IN+UN)
					// */
					// do {
					// NEItemIdx = Randoms.uniform(numItems);
					// if (!INitemSet.contains(NEItemIdx) && !UNitemSet.contains(NEItemIdx))
					// break;
					// } while (true);

					/*
					 * |IN|=|UN|, NE = S\(IN)
					 */
					do {
						NEItemIdx = Randoms.uniform(numItems);
						if (!INitemSet.contains(NEItemIdx) && !UNitemSet.contains(NEItemIdx))
							break;
					} while (true);

					break;

				}

				// update parameters
				double INPredictRating = predict(userIdx, INItemIdx);
				double NEPredictRating = predict(userIdx, NEItemIdx);
				double UNPredictRating = predict(userIdx, UNItemIdx);

				double diffValueA = (INPredictRating - NEPredictRating) * alpha;
				double diffValueB = (INPredictRating - UNPredictRating) * beta;
				// double diffValueC = (NEPredictRating - UNPredictRating) * gamma;

				double lossValue = -Math.log(Maths.logistic(diffValueA)) - Math.log(Maths.logistic(diffValueB));
				loss += lossValue;

				double deriValueA = Maths.logistic(-diffValueA);
				double deriValueB = Maths.logistic(-diffValueB);
				// double deriValueC = Maths.logistic(-diffValueC);

				// // update bi, bj
				// double INBiasValue = itemBiases.get(INItemIdx);
				// double regBias = 0.01;
				// itemBiases.add(INItemIdx, learnRate * (alpha * deriValueA + beta * deriValueB
				// - regBias * INBiasValue));
				//
				// double NEBiasValue = itemBiases.get(NEItemIdx);
				// itemBiases.add(NEItemIdx, learnRate * (-deriValueA * alpha - regBias *
				// NEBiasValue));
				//
				// double UNBiasValue = itemBiases.get(UNItemIdx);
				// itemBiases.add(UNItemIdx, learnRate * (-deriValueB * beta - regBias *
				// UNBiasValue));

				for (int factorIdx = 0; factorIdx < numFactors; factorIdx++) {
					double userFactorValue = userFactors.get(userIdx, factorIdx);
					double INItemFactorValue = itemFactors.get(INItemIdx, factorIdx);
					double NEItemFactorValue = itemFactors.get(NEItemIdx, factorIdx);
					double UNItemFactorValue = itemFactors.get(UNItemIdx, factorIdx);

					userFactors.add(userIdx, factorIdx,
							learnRate * (deriValueA * alpha * (INItemFactorValue - NEItemFactorValue)
									+ deriValueB * beta * (INItemFactorValue - UNItemFactorValue)
									- regUser * userFactorValue));
					itemFactors.add(INItemIdx, factorIdx, learnRate * (deriValueA * alpha * userFactorValue
							+ deriValueB * beta * userFactorValue - regItem * INItemFactorValue));
					itemFactors.add(NEItemIdx, factorIdx,
							learnRate * (deriValueA * (-alpha) * userFactorValue - regItem * NEItemFactorValue));
					itemFactors.add(UNItemIdx, factorIdx,
							learnRate * (deriValueB * (-beta) * userFactorValue - regItem * UNItemFactorValue));

					loss += regUser * userFactorValue * userFactorValue
							+ regItem * INItemFactorValue * INItemFactorValue
							+ regItem * NEItemFactorValue * NEItemFactorValue
							+ regItem * UNItemFactorValue * UNItemFactorValue;

					// if (indicator == 0) {
					// itemFactors.add(UNItemIdx, factorIdx,
					// learnRate * (deriValue * -(beta) * userFactorValue - regItem *
					// UNItemFactorValue));
					// loss += regUser * UNItemFactorValue * UNItemFactorValue;
					// }
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

		return new HashSet(sparseMatrix.getColumns_NE(userIdx, N));
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