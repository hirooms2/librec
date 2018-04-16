package net.librec.data.model;

public class Pair implements Comparable<Pair> {

	int left;
	double right;

	public Pair(int left, double val) {
		this.left = left;
		this.right = val;
	}

	@Override
	public int compareTo(Pair o) {
		// TODO Auto-generated method stub
		if (this.right < o.right)
			return -1;
		else if (this.right == o.right)
			return 0;
		return 1;
	}

	public int getLeft() {
		return left;
	}

	public double getRight() {
		return right;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return String.valueOf(right);
	}

}