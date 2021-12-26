package pkg;

import org.slf4j.Logger;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.impl.buffer.nio.NioDataBufferFactory;
import org.tensorflow.types.TInt32;

import java.nio.IntBuffer;
import java.util.Arrays;

import static org.slf4j.LoggerFactory.getLogger;

public class ModelLoader {
    private static final Logger logger = getLogger(ModelLoader.class);
    private final String modelPath;
    private final String tag;

    static {
        TensorFlow.loadLibrary("binary/_cuckoo_hashtable_ops.so");
    }

    public ModelLoader(String path, String tag) {
        this.modelPath = path;
        this.tag = tag;
    }

    public void load() {
        SavedModelBundle savedModelBundle = SavedModelBundle.load(this.modelPath, this.tag);

        int pa[][] = new int[1][512];
        Arrays.fill(pa[0], 1);

        TInt32 tensorPA = TInt32.tensorOf(Shape.of(1, 512));
        StdArrays.copyTo(pa, tensorPA);
        Tensor tensor = savedModelBundle.session().runner()
                .feed("pa", tensorPA)
                .feed("pb", tensorPA)
                .fetch("Add:0")
                .run().get(0);

        logger.info("ret : {}", tensor.asRawTensor());
        TInt32 result = (TInt32) tensor;
        int returnValue[] = new int[8];
        IntDataBuffer intDataBuffer = NioDataBufferFactory.create(IntBuffer.wrap(returnValue));
        result.read(intDataBuffer);
        logger.info("ret: {}", returnValue);
    }

    public static void main(String[] args) {
        String p = args[0];
        String tag = args[1];
        ModelLoader modelLoader = new ModelLoader(p, tag);
        modelLoader.load();
    }
}
