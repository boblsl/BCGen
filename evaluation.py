import nltk
from nltk.translate.bleu_score import *
# import metrics
import sys
# from rouge import Rouge
# from nltk.translate.meteor_score import meteor_score

def nltk_bleu(hypotheses, references, output_path, output_path1):
    output_file = open(output_path, 'w', encoding='utf-8')
    output_file1 = open(output_path1, 'w', encoding='utf-8')
    refs = []
    count = 0
    total_score = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        s = "y: " + ref + "\n" + "hyp: " + hyp + "\n"
        # remove the '< beg >' and '< end >'
        hyp = hyp.split()[3:-3]
        ref = ref.split()[3:-3]
        Cumulate_1_gram = 0
        Cumulate_2_gram = 0
        Cumulate_3_gram = 0
        Cumulate_4_gram = 0
        score = 0.0
        if len(hyp) >= 4:
            try:
                score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method3)
                Cumulate_1_gram = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method3)
                Cumulate_2_gram = sentence_bleu([ref], hyp, weights=(0, 1, 0, 0), smoothing_function=cc.method3)
                Cumulate_3_gram = sentence_bleu([ref], hyp, weights=(0, 0, 1, 0), smoothing_function=cc.method3)
                Cumulate_4_gram = sentence_bleu([ref], hyp, weights=(0, 0, 0, 1), smoothing_function=cc.method3)

            except Exception as ex:
                print("==ERROR==\n", ex)
                print("=========")
                exit(0)

        total_score += score
        total_bleu1 += Cumulate_1_gram
        total_bleu2 += Cumulate_2_gram
        total_bleu3 += Cumulate_3_gram
        total_bleu4 += Cumulate_4_gram

        count += 1
        s = s + "bleu-4,mothod3: " + str(score) + "\n"
        s = s + "Cumulate_1_gram: " + str(Cumulate_1_gram) + "\n"
        s = s + "Cumulate_2_gram: " + str(Cumulate_2_gram) + "\n"
        s = s + "Cumulate_3_gram: " + str(Cumulate_3_gram) + "\n"
        s = s + "Cumulate_4_gram: " + str(Cumulate_4_gram) + "\n======================\n\n"
        output_file.write(s)
        output_file1.write(str(score) + '\n')

    # output BLEU：
    avg_score = total_score / count
    print ('avg_score: %.4f' % avg_score)
    avg_bleu1 = total_bleu1 / count
    print('avg_bleu1: %.4f' % avg_bleu1)
    avg_bleu2 = total_bleu2 / count
    print('avg_bleu2: %.4f' % avg_bleu2)
    avg_bleu3 = total_bleu3 / count
    print('avg_bleu3: %.4f' % avg_bleu3)
    avg_bleu4 = total_bleu4 / count
    print('avg_bleu4: %.4f' % avg_bleu4)

    output_file.close()
    return avg_score


def nltk_bleu_old(hypotheses, references, output_path):
    output_file = open(output_path, 'w', encoding='utf-8')
    refs = []
    count = 0
    total_score = 0.0
    total_bleu1 = 0.0
    total_bleu2 = 0.0
    total_bleu3 = 0.0
    total_bleu4 = 0.0
    # total_rougeL = 0.0
    # total_meteor = 0.0

    cc = SmoothingFunction()

    for hyp, ref in zip(hypotheses, references):
        # if count >= 0 and count < 20:
        #     print("y: ", ref)
        #     print("hyp: ", hyp)
        #     print("==================")
        s = "y: " + ref + "\n" + "hyp: " + hyp + "\n"
        hyp = hyp.split()
        ref = ref.split()
        Cumulate_1_gram = 0
        Cumulate_2_gram = 0
        Cumulate_3_gram = 0
        Cumulate_4_gram = 0
        score = 0.0
        if len(hyp) >= 4:
            try:
                score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method3)
                Cumulate_1_gram = sentence_bleu([ref], hyp, weights=(1, 0, 0, 0), smoothing_function=cc.method3)
                Cumulate_2_gram = sentence_bleu([ref], hyp, weights=(0, 1, 0, 0), smoothing_function=cc.method3)
                Cumulate_3_gram = sentence_bleu([ref], hyp, weights=(0, 0, 1, 0), smoothing_function=cc.method3)
                Cumulate_4_gram = sentence_bleu([ref], hyp, weights=(0, 0, 0, 1), smoothing_function=cc.method3)
                # print("get rouge")
                # rouge = Rouge()
                # rouge_score = rouge.get_scores(ref, hyp)
                #
                # print("get rouge success!")
                '''
                f:F1值  p：查准率  R：召回率
                '''
                # print("ROUGE")
                # print(rouge_score[0]["rouge-1"])
                # print(rouge_score[0]["rouge-2"])
                # print(rouge_score[0]["rouge-l"]['r'])

                # meteor = round(meteor_score([ref], hyp), 4)

                # print('Cumulate 1-gram :%f' \
                #       % sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),smoothing_function=cc.method3))
                # print('Cumulate 2-gram :%f' \
                #       % sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0),smoothing_function=cc.method3))
                # print('Cumulate 3-gram :%f' \
                #       % sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0),smoothing_function=cc.method3))
                # print('Cumulate 4-gram :%f' \
                #       % sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method3))

                # print('method1 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method1))
                # print('method2 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method2))
                # print('method3 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method3))
                # print('method4 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method4))
                # print('method5 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method5))
                # print('method6 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method6))
                # print('method7 :%f' \
                #       % sentence_bleu([ref], hyp, smoothing_function=cc.method7))

            except Exception as ex:
                print("==ERROR==\n", ex)
                print("=========")
                exit(0)

        # score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
        # if count >= 0 and count < 20:
        #     print("@.@ bleu-4,mothod4: ", score)
        total_score += score
        total_bleu1 += Cumulate_1_gram
        total_bleu2 += Cumulate_2_gram
        total_bleu3 += Cumulate_3_gram
        total_bleu4 += Cumulate_4_gram
        # total_rougeL += rouge_score[0]["rouge-l"]['r']
        # total_meteor += meteor

        count += 1
        s = s + "bleu-4,mothod3: " + str(score) + "\n"
        s = s + "Cumulate_1_gram: " + str(Cumulate_1_gram) + "\n"
        s = s + "Cumulate_2_gram: " + str(Cumulate_2_gram) + "\n"
        s = s + "Cumulate_3_gram: " + str(Cumulate_3_gram) + "\n"
        s = s + "Cumulate_4_gram: " + str(Cumulate_4_gram) + "\n======================\n\n"
        # s = s + "rougeL: " + str(rouge_score[0]["rouge-l"]['r']) + "\n======================\n\n"
        # s = s + "meteor: " + str(meteor) + "\n======================\n\n"
        output_file.write(s)

    avg_score = total_score / count
    print ('avg_score: %.4f' % avg_score)
    avg_bleu1 = total_bleu1 / count
    print('avg_bleu1: %.4f' % avg_bleu1)
    avg_bleu2 = total_bleu2 / count
    print('avg_bleu2: %.4f' % avg_bleu2)
    avg_bleu3 = total_bleu3 / count
    print('avg_bleu3: %.4f' % avg_bleu3)
    avg_bleu4 = total_bleu4 / count
    print('avg_bleu4: %.4f' % avg_bleu4)

    output_file.close()
    return avg_score


def evaluate(reference, predictions, output_path, output_path1):
    hypotheses = []
    print('start evaluation')
    with open(predictions, 'r') as file:
        for line in file:
            hypotheses.append(line.strip())

    references = []
    with open(reference, 'r') as file:
        for line in file:
            references.append(line.strip())

    return nltk_bleu(hypotheses, references, output_path, output_path1)


if __name__ == '__main__':
    # evaluate(sys.argv[1], sys.argv[1])
    evaluate('./output/refs_wash.txt', './output/hyps_wash.txt', './output/result000.txt', './output/result001.txt')
