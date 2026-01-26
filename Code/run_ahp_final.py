from ahp_utils import ahp_consistency
import pipeline

def report(title, criteria, matrix):
    w, lam, ci, cr = ahp_consistency(matrix, method="eigen")
    print("=" * 70)
    print(title)
    print("Criteria:", criteria)
    print("Weights :", [round(x, 6) for x in w])
    print("lambda_max:", round(lam, 6))
    print("CI       :", round(ci, 6))
    print("CR       :", round(cr, 6))

if __name__ == "__main__":
    report("Isfahan", pipeline.criteria_I, pipeline.pairwise_I)
    report("Yazd", pipeline.criteria_Y, pipeline.pairwise_Y)
    report("Government", pipeline.criteria_G, pipeline.pairwise_G)




