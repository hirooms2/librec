/**
 * Copyright (C) 2016 LibRec
 *
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.recommender.cf.ranking;

import net.librec.BaseTestCase;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

/**
 * BPR Test Case corresponds to BPRRecommender
 * {@link net.librec.recommender.cf.ranking.BPRRecommender}
 *
 * @author SunYatong
 */
public class MyRecTestCase extends BaseTestCase {
	@Override
	@Before
	public void setUp() throws Exception {
		super.setUp();
	}

	/**
	 * test the whole process of BPR recommendation
	 *
	 * @throws ClassNotFoundException
	 * @throws LibrecException
	 * @throws IOException
	 */
	@Test
	public void testRecommender() throws ClassNotFoundException, LibrecException, IOException {
		Configuration.Resource resource = new Configuration.Resource("rec/cf/ranking/myrec-test.properties");
		conf.addResource(resource);
		RecommenderJob job = new RecommenderJob(conf);

		for (double alpha = 0; alpha <= 100; alpha += 1) {
			for (double beta = 0; beta <= 100; beta += 1) {
				if (alpha + beta == 100) {
					job.setAlpha(alpha / 100);
					job.setBeta(beta / 100);
					job.setGamma(0);
					job.runJob();
				}
			}
		}

	}
}
