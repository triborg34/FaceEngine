    #        # self.db_queue = Queue(maxsize=1000)  # Limit queue size
        # self.db_task = None
        # self.stats = {
        #     'processed': 0,
        #     'queued': 0,
        #     'failed': 0
        # }
    # async def start_background_processing(self):
    #     """Start the background database processing task"""
    #     logging.info("Start the background database processing task")
    #     self.db_task = asyncio.create_task(self._process_db_queue())

    # async def stop_background_processing(self):
    #     """Gracefully stop background processing"""
    #     logging.info("Gracefully stop background processing")
    #     if self.db_task:
    #         self.db_task.cancel()
    #         try:
    #             await self.db_task
    #         except asyncio.CancelledError:
    #             pass

    # async def _process_db_queue(self):
    #     """Background task to process database insertions"""
    #     while True:
    #         try:
    #             # Get item from queue (blocks if empty)
    #             data = await self.db_queue.get()

    #             try:
    #                 await insertToDb(
    #                     data['name'],
    #                     data['frame'],
    #                     data['face'],
    #                     data['human_crop'],
    #                     data['score'],
    #                     data['track_id'],
    #                     data['gender'],
    #                     data['age'],
    #                     data['role'],
    #                     data['path']
    #                 )
    #                 self.stats['processed'] += 1

    #             except Exception as e:
    #                 logging.error(f"Error inserting to DB: {e}")
    #                 self.stats['failed'] += 1

    #             finally:
    #                 # Mark task as done
    #                 self.db_queue.task_done()

    #         except asyncio.CancelledError:
    #             break
    #         except Exception as e:
    #             logging.error(f"Unexpected error in DB queue processor: {e}")
    #             await asyncio.sleep(1)  # Prevent tight error loops

    # async def queue_db_insertion(self, name, frame, face, human_crop,
    #                              score, track_id, gender, age, role, path):
    #     """Queue a database insertion (non-blocking)"""
    #     try:
    #         # Create a copy of image data to avoid reference issues
    #         data = {
    #             'name': name,
    #             'frame': frame.copy(),  # Important: copy arrays
    #             'face': face.copy(),
    #             'human_crop': human_crop.copy(),
    #             'score': score,
    #             'track_id': track_id,
    #             'gender': gender,
    #             'age': age,
    #             'role': role,
    #             'path': path
    #         }

    #         # Try to add to queue (non-blocking)
    #         self.db_queue.put_nowait(data)
    #         self.stats['queued'] += 1

    #     except asyncio.QueueFull:
    #         logging.warning("Database queue is full, dropping frame data")
    #         # Optionally: implement a strategy like dropping oldest items
    #     except Exception as e:
    #         logging.error(f"Error queuing DB insertion: {e}")

    # def get_queue_stats(self):
    #     """Get current queue statistics"""
    #     return {
    #         **self.stats,
    #         'queue_size': self.db_queue.qsize(),
    #         'queue_full': self.db_queue.full()
    #     }
    
    
