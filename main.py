from util import word2vec, load_all_data
from model.seq2seq import Seq2seq
import json


def train():
    batch_size = 64
    epochs = 3
    steps_per_epoch = 10
    lr = 1e-3
    char_size = 128
    z_dim = 128
    base_path = './data/今日头条新闻数据/'
    titles, texts = load_all_data(base_path)
    # 此时语料加载完毕。
    print("标题个数:", len(titles))  # 标题个数: 10398
    print("语料个数:", len(texts))  # 语料个数: 10398
    chars, id2char, char2id = word2vec(titles, texts)
    print("开始训练")
    S = Seq2seq(chars=chars, char2id=char2id, id2char=id2char, char_size=char_size, z_dim=z_dim)
    S.train(X_train=texts, Y_train=titles, epochs=epochs, batch_size=batch_size, lr=lr, steps_per_epoch=steps_per_epoch)


def predict():
    char_size = 128
    z_dim = 128
    base_path = './data/今日头条新闻数据/'
    titles, texts = load_all_data(base_path)
    s1 = '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
    s2 = '8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    chars, id2char, char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i): j for i, j in id2char.items()}
    S = Seq2seq(chars=chars, char2id=char2id, id2char=id2char, char_size=char_size, z_dim=z_dim)
    title_pre = S.predict(s=texts[4], model_weights='./best_model.weights')
    print('title_pre=',title_pre)
    print('title_true',titles[4])
    return title_pre


if __name__ == '__main__':
    predict()
    # train()
