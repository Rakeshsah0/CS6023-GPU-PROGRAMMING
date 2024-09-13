%%writefile /content/drive/MyDrive/GPUA3/submit/main.cu
/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

__global__ void applyTranslateKernal(int numberOfTranslation,int* translationsValues,int *deviceShiftedX,int *deviceShiftedY){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<numberOfTranslation){
		int messNumber= translationsValues[i*3];
		int taskNumber=translationsValues[i*3+1];
		int value=translationsValues[i*3+2];
		switch(taskNumber){
			case 0:
				atomicAdd(&deviceShiftedX[messNumber],-value);
				break;
			case 1:
				atomicAdd(&deviceShiftedX[messNumber],value);
				break;
			case 2:
				atomicAdd(&deviceShiftedY[messNumber],-value);
				break;
			case 3:
				atomicAdd(&deviceShiftedY[messNumber],value);
				break;

		}
	}
}

__device__ void bfsKernal(int n,int E,int V,int* deviceShiftedX,int* deviceShiftedY,int* deviceGlobalCoordinatesX,int* deviceGlobalCoordinatesY,int* deviceOffset,int* deviceCSR){
	int* queue=(int*)malloc(sizeof(int)*V);
	int front = 0, rear = 0;
	queue[rear++] = n;
	atomicAdd(&deviceGlobalCoordinatesX[n], deviceShiftedX[n]);
	atomicAdd(&deviceGlobalCoordinatesY[n], deviceShiftedY[n]);
	while (front < rear) {
			int current_vertex = queue[front++];
			int start = deviceOffset[current_vertex];
			int end = (current_vertex == V - 1) ? E : deviceOffset[current_vertex + 1];

			for (int i = start; i < end; ++i) {
					int neighbor = deviceCSR[i];
					queue[rear++] = neighbor;

					// Apply shifts to global coordinates of neighbor
					atomicAdd(&deviceGlobalCoordinatesX[neighbor], deviceShiftedX[n]);
					atomicAdd(&deviceGlobalCoordinatesY[neighbor], deviceShiftedY[n]);
			}
	}

	free(queue);
}


__global__ void applyTransitiveKernal(int E,int V,int* deviceShiftedX,int* deviceShiftedY,int* deviceGlobalCoordinatesX,int* deviceGlobalCoordinatesY,int* deviceOffset,int* deviceCSR){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<V){
		bfsKernal(i,E,V,deviceShiftedX,deviceShiftedY,deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,deviceOffset,deviceCSR);
	}
	__syncthreads();
}


__global__ void opacityMatrixKernal(int V,int* resultOpacity,int* deviceOpacity,int* deviceFrameSizeX,int* deviceFrameSizeY,int* deviceGlobalCoordinatesX,int* deviceGlobalCoordinatesY,int frameSizeX,int frameSizeY) // launch no. of nodes
{
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<V)
	{
			for(int i=0;i<deviceFrameSizeX[idx];i++)
			{
				for(int j=0;j<deviceFrameSizeY[idx];j++)
				{
					int row=i+deviceGlobalCoordinatesX[idx];
					int col =j +deviceGlobalCoordinatesY[idx];
					if(row >= 0 && row<frameSizeX && col >= 0 && col<frameSizeY )
					{
						int index = row * frameSizeY + col;
						//resultOpacity[index] = deviceOpacity[idx];
						atomicMax(&resultOpacity[index],deviceOpacity[idx]);
						
					}
				}
			}
	}
}


__global__ void mapKernal(int V,int* resultOpacity,int* deviceArr){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d\n",idx);
	if(idx<V)
	{
		deviceArr[resultOpacity[idx]]=idx;
		//printf("%d ",resultOpacity[idx]);
	}

}




__global__ void scene_creation(int frameSizeX,int frameSizeY,int* resultOpacity,int* deviceArr,int* deviceGlobalCoordinatesX,int* deviceGlobalCoordinatesY,int** deviceMess,int* deviceFrameSizeX,int *deviceFrameSizeY){
	int idx=blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<frameSizeX*frameSizeY){
		if(resultOpacity[idx]==-1)
		{
			resultOpacity[idx]=0;
		}
		else

		{
			int node = deviceArr[resultOpacity[idx]];
			int x= idx /frameSizeY;
			int y=idx% frameSizeY;
			int row = x-deviceGlobalCoordinatesX[node];
			int col = y-deviceGlobalCoordinatesY[node];
			resultOpacity[idx]=deviceMess[node][row*deviceFrameSizeY[node]+col];
		}
	}
}
/*
*/


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

  int* hostTranslationsValues=(int*)malloc(sizeof(int)*numTranslations*3);
	for(int i=0;i<numTranslations;i++){
		hostTranslationsValues[i*3]=translations[i][0];
		hostTranslationsValues[i*3+1]=translations[i][1];
		hostTranslationsValues[i*3+2]=translations[i][2];
	}

  int *deviceOpacity,*deviceFrameSizeX,*deviceFrameSizeY,*deviceGlobalCoordinatesX,*deviceGlobalCoordinatesY,*deviceShiftedX,*deviceShiftedY,*deviceTranslationsValues;
  cudaMalloc(&deviceOpacity, sizeof(int) * V);
  cudaMalloc(&deviceFrameSizeX, sizeof(int) * V);
  cudaMalloc(&deviceFrameSizeY, sizeof(int) * V);
  cudaMalloc(&deviceGlobalCoordinatesX, sizeof(int) * V);
  cudaMalloc(&deviceGlobalCoordinatesY, sizeof(int) * V);
  cudaMalloc(&deviceShiftedX, sizeof(int) * V);
  cudaMalloc(&deviceShiftedY, sizeof(int) * V);
  cudaMalloc(&deviceTranslationsValues, sizeof(int) * 3*numTranslations);

  cudaMemcpy(deviceOpacity,hOpacity,sizeof(int)*V,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFrameSizeX,hFrameSizeX,sizeof(int)*V,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceFrameSizeY,hFrameSizeY,sizeof(int)*V,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceGlobalCoordinatesX,hGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceGlobalCoordinatesY,hGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyHostToDevice);
	cudaMemset(deviceShiftedX,0,V);
	cudaMemset(deviceShiftedY,0,V);
  cudaMemcpy(deviceTranslationsValues,hostTranslationsValues,sizeof(int)*3*numTranslations,cudaMemcpyHostToDevice);


	int **deviceMess;
	cudaMalloc(&deviceMess,sizeof(int*)*V);
	int **Mess = (int **) malloc(sizeof(int *)*V);
	for(int i=0;i<V;i++){
		int *newMess;
		cudaMalloc(&newMess,sizeof(int)*hFrameSizeX[i]*hFrameSizeY[i]);
		cudaMemcpy(newMess,hMesh[i],sizeof(int)*hFrameSizeX[i]*hFrameSizeY[i],cudaMemcpyHostToDevice);
		Mess[i]=newMess;
	}
	cudaMemcpy(deviceMess,Mess,sizeof(int*)*V,cudaMemcpyHostToDevice);

	int blocksNum=ceil(numTranslations*1.0/1024);
	applyTranslateKernal<<<blocksNum,1024>>>(numTranslations,deviceTranslationsValues,deviceShiftedX,deviceShiftedY);
  cudaDeviceSynchronize();


	int* deviceOffset,*deviceCSR;
	cudaMalloc(&deviceOffset,sizeof(int)*(V+1));
	cudaMalloc(&deviceCSR,sizeof(int)*E);
	cudaMemcpy(deviceOffset,hOffset,sizeof(int)*(V+1),cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCSR,hCsr,sizeof(int)*E,cudaMemcpyHostToDevice);

	blocksNum=ceil(V*1.0/1024);
	applyTransitiveKernal<<<blocksNum,1024>>>(E,V,deviceShiftedX,deviceShiftedY,deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,deviceOffset,deviceCSR);
	cudaDeviceSynchronize();
/*	cudaMemcpy(hGlobalCoordinatesX,deviceGlobalCoordinatesX,sizeof(int)*V,cudaMemcpyDeviceToHost);
  cudaMemcpy(hGlobalCoordinatesY,deviceGlobalCoordinatesY,sizeof(int)*V,cudaMemcpyDeviceToHost);

	for(int i=0;i<V;++i)
	{
		printf("%d %d\n",hGlobalCoordinatesX[i],hGlobalCoordinatesY[i]);
	}
	*/
	

	int* resultOpacity;
	cudaMalloc(&resultOpacity,sizeof(int)*frameSizeX*frameSizeY);
	cudaMemset(resultOpacity,-1,sizeof(int)*frameSizeX*frameSizeY);
	blocksNum=ceil(V*1.0/1024);
	opacityMatrixKernal<<<blocksNum,1024>>>(V,resultOpacity,deviceOpacity,deviceFrameSizeX,deviceFrameSizeY,deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,frameSizeX,frameSizeY);
	cudaDeviceSynchronize();


	int* deviceArr;
	cudaMalloc(&deviceArr,sizeof(int)*(300000001));
	blocksNum=ceil(V*1.0/1024);
	mapKernal<<<blocksNum,1024>>>(V,deviceOpacity,deviceArr);
	cudaDeviceSynchronize();


/*
	int* harr=(int*)malloc(sizeof(int)*(300000001));
	cudaMemcpy(harr,deviceArr,sizeof(int)*(300000001),cudaMemcpyDeviceToHost);
	for(int i=0;i<V;i++){
		printf("%d %d\n",hOpacity[i],harr[hOpacity[i]]);
	}
	*/


	blocksNum=ceil(frameSizeX*frameSizeY*1.0/1024);
	scene_creation<<<blocksNum,1024>>>(frameSizeX,frameSizeY,resultOpacity,deviceArr,deviceGlobalCoordinatesX,deviceGlobalCoordinatesY,deviceMess,deviceFrameSizeX,deviceFrameSizeY);

	/*
	
*/
	for (int i = 0; i < V; i++) {
    cudaFree(Mess[i]);
	}
	cudaFree(Mess);
	cudaFree(deviceMess);
	cudaFree(deviceTranslationsValues);
	cudaFree(deviceArr);
	cudaFree(hostTranslationsValues);
	cudaFree(deviceOpacity);
	cudaFree(deviceFrameSizeX);
	cudaFree(deviceFrameSizeY);
	cudaFree(deviceGlobalCoordinatesX);
	cudaFree(deviceGlobalCoordinatesY);
	cudaFree(deviceShiftedX);
	cudaFree(deviceShiftedY);
	cudaFree(deviceOffset);
	cudaFree(deviceCSR);
	cudaFree(resultOpacity);

  cudaMemcpy(hFinalPng,resultOpacity,sizeof(int)*frameSizeX*frameSizeY,cudaMemcpyDeviceToHost);
	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;

	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
