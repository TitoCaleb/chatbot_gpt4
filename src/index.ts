import { ChatOpenAI, OpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Client } from "@notionhq/client";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CombinedMemory, VectorStoreRetrieverMemory } from "langchain/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { LLMChain } from "langchain/chains";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

const model = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  //modelName: "gpt-4-turbo",
  modelName: "gpt-3.5-turbo",
});

//DevInconsistenciaGPT

const getInstructions = async () => {
  const notion = new Client({
    auth: process.env.NOTION_API_KEY,
  });

  const notionPages = await notion.blocks.children.list({
    block_id: process.env.NOTION_INCONSISTENCY_INSTRUCTIONS_ID || "",
  });

  const text = notionPages.results
    ?.map((block: any) => {
      let paragraph = "";
      if (block.type === "paragraph") {
        paragraph = block?.paragraph?.rich_text
          .map((rt: any) => rt.plain_text)
          .join("\n");
      }
      return paragraph;
    })
    .join("\n\n");
  return text;
};

const getEjemplo = async () => {
  const loader = new TextLoader("./ejemplo.txt");
  const docs = await loader.load();
  return docs;
};

const getEjemploOnboarding = async () => {
  const loader = new TextLoader("./ejemploOnboarding.txt");
  const docs = await loader.load();
  return docs;
};

const doAQuestion = async (question: string) => {
  const res = await model.invoke(question);
  console.log(res.content);
};

const addQuestion = async (question: string) => {
  // Crear vectores
  const retriever = await MemoryVectorStore.fromTexts(
    ["Caleb", "Castro", "trabaja", "en", "Blum SAF"],
    [{ id: 2 }, { id: 1 }, { id: 3 }, { id: 4 }, { id: 5 }],
    new OpenAIEmbeddings()
  );

  // Crear vectores desde un texto
  // const retriever = await MemoryVectorStore.fromDocuments(
  //   await getEjemplo(),
  //   new OpenAIEmbeddings()
  // );

  // Load documents
  const instruction = await getInstructions();

  /* console.log(instruction); */

  const memoryInitial = new VectorStoreRetrieverMemory({
    vectorStoreRetriever: retriever.asRetriever(6),
    memoryKey: "history_initial",
  });

  //   memoryInitial.saveContext(
  //     { input: instruction },
  //     { output: "Instrucciones" }
  //   );

  //Crear estructura del prompt
  const prompt = ChatPromptTemplate.fromTemplate(`
      Conversación inicial:
      {history_initial}

      Siguiente pregunta
      Usuario: {input}
      Respuesta IA:`);

  const chain = new LLMChain({
    llm: model,
    memory: new CombinedMemory({
      memories: [memoryInitial],
    }),
    prompt,
  });

  const res = await chain.invoke({ input: question });
  console.log(res.text);
};

const newAddQuestion = async (question: string) => {
  console.time("Tiempo de respuesta");
  // Crear vectores

  const store = await MemoryVectorStore.fromTexts(
    ["Caleb", "Castro", "trabaja", "en", "Blum SAF"],
    [{ id: 2 }, { id: 1 }, { id: 3 }, { id: 4 }, { id: 5 }],
    new OpenAIEmbeddings()
  );

  // const store = await MemoryVectorStore.fromDocuments(
  //   await getEjemploOnboarding(),
  //   new OpenAIEmbeddings()
  // );

  const instruction = await getInstructions();

  //Crear estructura del prompt
  const prompt = ChatPromptTemplate.fromTemplate(`
    Conversación inicial:
    {history_initial}

    {context}

    Siguiente pregunta
    Usuario: {input}
    Respuesta IA:`);

  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  const chain = await createRetrievalChain({
    retriever: store.asRetriever(6),
    combineDocsChain,
  });

  //   const runnable = RunnableSequence.from([prompt, model]);

  const res = await chain.invoke({
    input: question,
    history_initial: instruction,
  });

  console.log(res.answer);
  console.timeEnd("Tiempo de respuesta");
};

async function main() {
  // doAQuestion("Que es star wars?");
  // addQuestion("Quien es Caleb?");
  // addQuestion("Quien eres? Y que es star wars?");
  newAddQuestion("Como te llamas? Y para que empresa trabajas?");
}

main();
